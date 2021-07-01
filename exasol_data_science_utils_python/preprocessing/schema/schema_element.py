import unicodedata
from abc import ABC, abstractmethod


class UnicodeCategories:
    UPPERCASE_LETTER = 'Lu'
    LOWERCASE_LETTER = 'Ll'
    TITLECASE_LETTER = 'Lt'
    MODIFIER_LETTER = 'Lm'
    OTHER_LETTER = 'Lo'
    LETTER_NUMBER = 'Nl'
    NON_SPACING_MARK = 'Mn'
    COMBINING_SPACING_MARK = 'Mc'
    DECIMAL_DIGIT_NUMBER = 'Nd'
    CONNECTOR_PUNCTUATION = 'Pc'
    FORMAT = 'Cf'


# return (codePoint == 0x00B7);

class SchemaElement(ABC):

    def __init__(self, name: str):
        if not self.validate_name(name):
            raise ValueError(f"Name '{name}' is not valid")
        self.name = name

    def quoted_name(self):
        return f'"{self.name}"'

    @abstractmethod
    def fully_qualified(self) -> str:
        pass

    @classmethod
    def validate_name(self, name: str) -> bool:
        if name is None or name == "":
            return False
        if not self._validate_first_character(name[0]):
            return False
        for c in name[1:]:
            if not self._validate_follow_up_character(c):
                return False
        return True

    @classmethod
    def _validate_first_character(self, chararcter: str) -> bool:
        unicode_category = unicodedata.category(chararcter)
        return \
            unicode_category == UnicodeCategories.UPPERCASE_LETTER or \
            unicode_category == UnicodeCategories.LOWERCASE_LETTER or \
            unicode_category == UnicodeCategories.TITLECASE_LETTER or \
            unicode_category == UnicodeCategories.MODIFIER_LETTER or \
            unicode_category == UnicodeCategories.OTHER_LETTER or \
            unicode_category == UnicodeCategories.LETTER_NUMBER

    @classmethod
    def _validate_follow_up_character(self, chararcter: str) -> bool:
        unicode_category = unicodedata.category(chararcter)
        return \
            unicode_category == UnicodeCategories.UPPERCASE_LETTER or \
            unicode_category == UnicodeCategories.LOWERCASE_LETTER or \
            unicode_category == UnicodeCategories.TITLECASE_LETTER or \
            unicode_category == UnicodeCategories.MODIFIER_LETTER or \
            unicode_category == UnicodeCategories.OTHER_LETTER or \
            unicode_category == UnicodeCategories.LETTER_NUMBER or \
            unicode_category == UnicodeCategories.NON_SPACING_MARK or \
            unicode_category == UnicodeCategories.COMBINING_SPACING_MARK or \
            unicode_category == UnicodeCategories.DECIMAL_DIGIT_NUMBER or \
            unicode_category == UnicodeCategories.CONNECTOR_PUNCTUATION or \
            unicode_category == UnicodeCategories.FORMAT or \
            chararcter == '\u00B7'
