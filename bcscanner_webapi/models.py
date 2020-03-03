import json

from django.db import models


class NERCandidates:

    def __init__(self, candidates=None, entity="unassigned"):

        if candidates is None:
            candidates = []

        self.candidates = candidates
        self.entity = entity

    def from_json(self, str):
        self.from_dict(json.loads(str))

    def from_dict(self, values):
        self.candidates = values['candidates'] if 'candidates' in values else []
        self.entity = values['entity'] if 'entity' in values else 'unassigned'

    def to_dict(self):
        return {
            'candidates': self.candidates,
            'entity': self.entity
        }

    def to_json(self):
        return json.dumps(self.to_dict())


class NERCandidatesField(models.Field):

    description = "A Named Entity field"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def parse_from_string(value):
        obj = NERCandidates()
        obj.from_json(value)
        return obj

    def from_db_value(self, value, expression, connection):
        if value is None:
            return value
        return self.parse_from_string(value)

    def to_python(self, value):
        if isinstance(value, NERCandidates):
            return value
        if value is None:
            return value
        return self.parse_from_string(value)

    def get_prep_value(self, value):
        return value.to_json()


class BCScanInfo(models.Model):

    names = NERCandidatesField()
    titles = NERCandidatesField()
    occupations = NERCandidatesField()
    organisations = NERCandidatesField()
    street = NERCandidatesField()
    city = NERCandidatesField()
    postal_code = NERCandidatesField()
    country = NERCandidatesField()
    phones = NERCandidatesField()
    phone_types = NERCandidatesField()
    www = NERCandidatesField()
    emails = NERCandidatesField()
    unassigned = NERCandidatesField()


class BCScanRecord(models.Model):

    scanned_image = models.FileField()
    scan_infos = models.ManyToManyField(BCScanInfo)
