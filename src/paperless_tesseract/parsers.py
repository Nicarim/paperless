import itertools
import os
import re
import subprocess
import tempfile
import time
from io import BytesIO
from multiprocessing.pool import Pool

import langdetect
import pdftotext
import pyocr
from PIL import Image
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import \
    TextOperationStatusCodes, ComputerVisionErrorException
from django.conf import settings
from msrest.authentication import CognitiveServicesCredentials
from pyocr.libtesseract.tesseract_raw import \
    TesseractError as OtherTesseractError
from pyocr.tesseract import TesseractError

from documents.parsers import DocumentParser, ParseError
from paperless.settings import OCR_AZURE_ENDPOINT, OCR_AZURE_KEY
from .languages import ISO639


class OCRError(Exception):
    pass


class RasterisedDocumentParser(DocumentParser):
    """
    This parser uses Tesseract to try and get some text out of a rasterised
    image, whether it's a PDF, or other graphical format (JPEG, TIFF, etc.)
    """

    CONVERT = settings.CONVERT_BINARY
    GHOSTSCRIPT = settings.GS_BINARY
    DENSITY = settings.CONVERT_DENSITY if settings.CONVERT_DENSITY else 300
    THREADS = int(settings.OCR_THREADS) if settings.OCR_THREADS else None
    UNPAPER = settings.UNPAPER_BINARY
    DEFAULT_OCR_LANGUAGE = settings.OCR_LANGUAGE
    OCR_ALWAYS = settings.OCR_ALWAYS

    def __init__(self, path):
        super().__init__(path)
        self._text = None

    def get_thumbnail(self):
        """
        The thumbnail of a PDF is just a 500px wide image of the first page.
        """

        out_path = os.path.join(self.tempdir, "convert.png")

        # Run convert to get a decent thumbnail
        try:
            run_convert(
                self.CONVERT,
                "-scale", "500x5000",
                "-alpha", "remove",
                "-strip", "-trim",
                "{}[0]".format(self.document_path),
                out_path
            )
        except ParseError:
            # if convert fails, fall back to extracting
            # the first PDF page as a PNG using Ghostscript
            self.log(
                "warning",
                "Thumbnail generation with ImageMagick failed, "
                "falling back to Ghostscript."
            )
            gs_out_path = os.path.join(self.tempdir, "gs_out.png")
            cmd = [self.GHOSTSCRIPT,
                   "-q",
                   "-sDEVICE=pngalpha",
                   "-o", gs_out_path,
                   self.document_path]
            if not subprocess.Popen(cmd).wait() == 0:
                raise ParseError("Thumbnail (gs) failed at {}".format(cmd))
            # then run convert on the output from gs
            run_convert(
                self.CONVERT,
                "-scale", "500x5000",
                "-alpha", "remove",
                "-strip", "-trim",
                gs_out_path,
                out_path
            )

        return out_path

    def _is_ocred(self):

        # Extract text from PDF using pdftotext
        text = get_text_from_pdf(self.document_path)

        # We assume, that a PDF with at least 50 characters contains text
        # (so no OCR required)
        return len(text) > 50

    def get_text(self):

        if self._text is not None:
            return self._text

        if not self.OCR_ALWAYS and self._is_ocred():
            self.log("info", "Skipping OCR, using Text from PDF")
            self._text = get_text_from_pdf(self.document_path)
            return self._text

        images = self._get_greyscale()

        try:
            self._text = self._get_ocr(images)
            return self._text
        except OCRError as e:
            raise ParseError(e)

    def _get_greyscale(self):
        """
        Greyscale images are easier for Tesseract to OCR
        """

        # Convert PDF to multiple PNMs
        pnm = os.path.join(self.tempdir, "convert-%04d.pnm")
        run_convert(
            self.CONVERT,
            "-density", str(self.DENSITY),
            "-depth", "8",
            "-type", "grayscale",
            self.document_path, pnm,
        )

        # Get a list of converted images
        pnms = []
        for f in os.listdir(self.tempdir):
            if f.endswith(".pnm"):
                pnms.append(os.path.join(self.tempdir, f))

        # Run unpaper in parallel on converted images
        with Pool(processes=self.THREADS) as pool:
            pool.map(run_unpaper, itertools.product([self.UNPAPER], pnms))

        # Return list of converted images, processed with unpaper
        pnms = []
        for f in os.listdir(self.tempdir):
            if f.endswith(".unpaper.pnm"):
                pnms.append(os.path.join(self.tempdir, f))

        return sorted(filter(lambda __: os.path.isfile(__), pnms))

    def _guess_language(self, text):
        try:
            guess = langdetect.detect(text)
            self.log("debug", "Language detected: {}".format(guess))
            return guess
        except Exception as e:
            self.log("warning", "Language detection error: {}".format(e))

    def _get_ocr(self, imgs):
        """
        Attempts to do the best job possible OCR'ing the document based on
        simple language detection trial & error.
        """

        if not imgs:
            raise OCRError("No images found")

        self.log("info", "OCRing the document")

        # Since the division gets rounded down by int, this calculation works
        # for every edge-case, i.e. 1
        middle = int(len(imgs) / 2)
        raw_text = self._ocr([imgs[middle]], self.DEFAULT_OCR_LANGUAGE)

        guessed_language = self._guess_language(raw_text)

        if not guessed_language or guessed_language not in ISO639:
            self.log("warning", "Language detection failed!")
            if settings.FORGIVING_OCR:
                self.log(
                    "warning",
                    "As FORGIVING_OCR is enabled, we're going to make the "
                    "best with what we have."
                )
                raw_text = self._assemble_ocr_sections(imgs, middle, raw_text)
                return raw_text
            error_msg = ("Language detection failed. Set "
                         "PAPERLESS_FORGIVING_OCR in config file to continue "
                         "anyway.")
            raise OCRError(error_msg)

        if ISO639[guessed_language] == self.DEFAULT_OCR_LANGUAGE:
            raw_text = self._assemble_ocr_sections(imgs, middle, raw_text)
            return raw_text

        try:
            return self._ocr(imgs, ISO639[guessed_language])
        except pyocr.pyocr.tesseract.TesseractError:
            if settings.FORGIVING_OCR:
                self.log(
                    "warning",
                    "OCR for {} failed, but we're going to stick with what "
                    "we've got since FORGIVING_OCR is enabled.".format(
                        guessed_language
                    )
                )
                raw_text = self._assemble_ocr_sections(imgs, middle, raw_text)
                return raw_text
            raise OCRError(
                "The guessed language ({}) is not available in this instance "
                "of Tesseract.".format(guessed_language)
            )

    def _ocr_from_azure_img(self, img, lang):
        pimg = Image.open(img)
        temp = BytesIO()
        with tempfile.TemporaryFile(dir='/tmp/') as f:
            pimg.save(f, format='png')
            f.seek(0)
            creds = CognitiveServicesCredentials(OCR_AZURE_KEY)
            cv_client = ComputerVisionClient(OCR_AZURE_ENDPOINT, creds)
            while True:
                try:
                    rp_res = cv_client.batch_read_file_in_stream(f, raw=True)
                    break
                except ComputerVisionErrorException as e:
                    if e.response.status_code == 429:
                        print(
                            "Exceeded quota for this minute, waiting 20 seconds...")
                        time.sleep(20)
                        continue
                    json_error = e.response.json()
                    if 'error' in json_error and 'code' in json_error['error']:
                        if json_error['error']['code'] == "InvalidImage":
                            self.log('error',
                                     'Azure couldn\'t recognize given image')
                            # TODO: Find out for what images does it happen
                            # TODO: We return instead of raising - let's just omit this error and mark it in a document
                            return '\n\n< invalid image >\n\n'

                    self.log('error',
                             'Response from azure is containing an error')
                    self.log('error', e.response.text)
                    time.sleep(60)
                    raise

        # Get the operation location (URL with an ID at the end) from the response
        operation_location_remote = rp_res.headers["Operation-Location"]
        # Grab the ID from the URL
        operation_id = operation_location_remote.split("/")[-1]
        while True:
            get_printed_text_results = cv_client.get_read_operation_result(
                operation_id)
            if get_printed_text_results.status not in ['NotStarted',
                                                       'Running']:
                break
            self.log('info', 'Waiting for Azure response...')
            time.sleep(5)
        text_response = []
        if get_printed_text_results.status == TextOperationStatusCodes.succeeded:
            for text_result in get_printed_text_results.recognition_results:
                for line in text_result.lines:
                    text_response.append(line.text)
        return '\n'.join(text_response)

    def _ocr(self, imgs, lang):
        """
        Performs a single OCR attempt.
        """

        if not imgs:
            return ""

        self.log("info", "Parsing for {}".format(lang))
        # TODO: USE AZURE LOL
        r = None
        if not OCR_AZURE_ENDPOINT:
            with Pool(processes=self.THREADS) as pool:
                r = pool.map(image_to_string, itertools.product(imgs, [lang]))
                r = " ".join(r)
        else:
            r = []
            for img in imgs:
                r.append(self._ocr_from_azure_img(img, lang))
            r = " ".join(r)

        # Strip out excess white space to allow matching to go smoother
        return strip_excess_whitespace(r)

    def _assemble_ocr_sections(self, imgs, middle, text):
        """
        Given a `middle` value and the text that middle page represents, we OCR
        the remainder of the document and return the whole thing.
        """
        text = self._ocr(imgs[:middle], self.DEFAULT_OCR_LANGUAGE) + text
        text += self._ocr(imgs[middle + 1:], self.DEFAULT_OCR_LANGUAGE)
        return text


def run_convert(*args):
    environment = os.environ.copy()
    if settings.CONVERT_MEMORY_LIMIT:
        environment["MAGICK_MEMORY_LIMIT"] = settings.CONVERT_MEMORY_LIMIT
    if settings.CONVERT_TMPDIR:
        environment["MAGICK_TMPDIR"] = settings.CONVERT_TMPDIR

    if not subprocess.Popen(args, env=environment).wait() == 0:
        raise ParseError("Convert failed at {}".format(args))


def run_unpaper(args):
    unpaper, pnm = args
    command_args = (unpaper, "--overwrite", pnm,
                    pnm.replace(".pnm", ".unpaper.pnm"))
    if not subprocess.Popen(command_args).wait() == 0:
        raise ParseError("Unpaper failed at {}".format(command_args))


def strip_excess_whitespace(text):
    collapsed_spaces = re.sub(r"([^\S\r\n]+)", " ", text)
    no_leading_whitespace = re.sub(
        r"([\n\r]+)([^\S\n\r]+)", '\\1', collapsed_spaces)
    no_trailing_whitespace = re.sub(
        r"([^\S\n\r]+)$", '', no_leading_whitespace)
    return no_trailing_whitespace


def image_to_string(args):
    img, lang = args
    ocr = pyocr.get_available_tools()[0]
    with Image.open(os.path.join(RasterisedDocumentParser.SCRATCH, img)) as f:
        if ocr.can_detect_orientation():
            try:
                orientation = ocr.detect_orientation(f, lang=lang)
                f = f.rotate(orientation["angle"], expand=1)
            except (TesseractError, OtherTesseractError, AttributeError):
                pass
        return ocr.image_to_string(f, lang=lang)


def get_text_from_pdf(pdf_file):
    with open(pdf_file, "rb") as f:
        try:
            pdf = pdftotext.PDF(f)
        except pdftotext.Error:
            return ""

    return "\n".join(pdf)
