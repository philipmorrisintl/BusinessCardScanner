import re
import os
import pickle
import numpy as np
from names_dataset import NameDataset

from .address_parser import parse_address
from .Config import Config


class NER:

    def __init__(self, config=None):

        if config is None:
            config = Config()

        self.link_regex = re.compile(r'(?:(?:(?:https?|ftp|file):\/\/)|www\.)(?:\S+(?::\S*)?@)?(?:(?!10(?:\.\d{1,3}){3}'
                                     r')(?!127(?:\.\d{1,3}){3})(?!169\.254(?:\.\d{1,3}){2})(?!192\.168(?:\.\d{1,3}){2})'
                                     r'(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])'
                                     r'(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))'
                                     r'|(?:(?:[a-z\\x{00a1}\-\\x{ffff}0-9]+-?)*[a-z\\x{00a1}\-\\x{ffff}0-9]+)(?:\.(?:[a'
                                     r'-z\\x{00a1}\-\\x{ffff}0-9]+-?)*[a-z\\x{00a1}\-\\x{ffff}0-9]+)*(?:\.(?:[a-z\\x{00'
                                     r'a1}\-\\x{ffff}]{2,})))(?::\d{2,5})?(?:\/[^\s]*)?')

        self.email_regex = re.compile(r'(([a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-'
                                      r'\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*"'
                                      r')@((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:'
                                      r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9]'
                                      r'[0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\['
                                      r'\x01-\x09\x0b\x0c\x0e-\x7f])+)\]))')

        self.phone_regex = re.compile(r'((?:(?:\(\s*(?:00|\+)([1-9][1-9][1-9]|[1-9][1-9]|[1-9])\s*\)\s*|(?:00|\+)([1-9]'
                                      r'[1-9][1-9]|[1-9][1-9]|[1-9])?\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]'
                                      r'1|[2-9][02-8][02-9]|[0-9][0-9][0-9]|[0-9][0-9]|[0-9])\s*\)|([0-9][1-9]|[0-9]1[0'
                                      r'2-9]|[2-9][02-8]1|[2-9][02-8][02-9]|[0-9][0-9][0-9]|[0-9][0-9]|[0-9]))??\s*(?:['
                                      r'.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2}|[0-9][0-9][0-9]|[0-9][0-9])'
                                      r'\s*(?:[.-]\s*)?((?:[0-9]\s*){4,8})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?)')

        self.names_lookup = NameDataset()

        self.org_occ_model = None
        model_path = config.ner_model
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "model_2cl.pkl")
        with open(model_path, "rb") as thefile:
            self.org_occ_model = pickle.load(thefile)

    def parse(self, lines):

        print("Lines:", lines)

        entities = []
        for iline, line in enumerate(lines):
            ents = self.find_entities(line, iline)
            entities += ents

        entities = self.cleanup_entities(lines, entities)

        entities = sorted(entities, key=lambda x: -x["prob"] if "prob" in x else 0)

        return entities

    def find_entities(self, line, iline):

        entities = []

        phone_records = self.find_phone_records(line, iline, entities)
        if phone_records is not None:
            entities += phone_records

        emails = self.find_emails(line, iline, entities)
        if emails is not None:
            entities += emails

        links = self.find_links(line, iline, entities)
        if links is not None:
            entities += links

        persons = self.find_persons(line, iline, entities)
        if persons is not None:
            entities += persons

        addresses = self.find_addresses(line, iline, entities)
        if addresses is not None:
            entities += addresses

        others = self.find_others(line, iline, entities)
        if others is not None:
            entities += others

        return entities

    def cleanup_entities(self, lines, entities):

        persons = []
        for entity in entities:
            if entity["entity"] == "person":
                persons += [entity]

        cleaned_persons = []
        if len(persons) > 1:
            for person in persons:
                n_from_same_line = 0
                for entity in entities:
                    if entity["line id"] == person["line id"]:
                        n_from_same_line += 1

                if n_from_same_line == 1:
                    cleaned_persons += [person]
                else:
                    person["entity"] = "unassigned"

        cleaned_entities = []
        if len(cleaned_persons) > 0:
            for entity in entities:
                if entity["entity"] != "person":
                    cleaned_entities += [entity]
            for person in cleaned_persons:
                cleaned_entities += [person]
        else:
            for entity in entities:
                cleaned_entities += [entity]

        return cleaned_entities

    def find_phone_records(self, line, iline, found_so_far):

        records = []
        results = self.phone_regex.findall(line)
        for result in results:
            records += [{
                "entity": "phone",
                "value": result[0],
                "line id": iline,
                "number": "".join(result[1:])
            }]
        return records

    def find_emails(self, line, iline, found_so_far):
        emails = []
        results = self.email_regex.findall(line)
        for result in results:
            emails += [{
                "entity": "e-mail",
                "value": result[0],
                "line id": iline,
                "user": result[1],
                "domain": result[2]
            }]
        return emails

    def find_links(self, line, iline, found_so_far):
        links = []
        results = self.link_regex.findall(line)
        for result in results:
            links += [{
                "entity": "www",
                "value": result,
                "line id": iline,
            }]
        return links

    def find_persons(self, line, iline, found_so_far):

        persons = []

        first_name = None
        last_name = None
        i_first_name = None
        i_last_name = None
        words = line.replace('.', ' ').replace('?', ' ').split(' ')
        for i_word, word in enumerate(words):

            is_first_name = self.names_lookup.search_first_name(word.lower())
            is_last_name = self.names_lookup.search_last_name(word.lower())

            if is_first_name and not is_last_name:
                if first_name is None:
                    first_name = word
                    i_first_name = i_word
            if not is_first_name and is_last_name:
                if last_name is None:
                    last_name = word
                    i_last_name = i_word
            if is_first_name and is_last_name:
                if first_name is None:
                    first_name = word
                    i_first_name = i_word
                elif last_name is None:
                    last_name = word
                    i_last_name = i_word

        if first_name is None and last_name is not None and len(words) > 1:
            for i in range(1, len(words)):
                if last_name == words[i]:
                    first_name = words[i-1]
                    i_first_name = i-1
        if last_name is None and first_name is not None and len(words) > 1:
            for i in range(len(words)-1):
                if first_name == words[i]:
                    last_name = words[i+1]
                    i_last_name = i+1

        if i_last_name is not None and i_first_name is not None:
            if i_last_name != i_first_name+1 and i_last_name != i_first_name-1:
                first_name = None
                last_name = None

        person = ""
        if first_name is not None:
            person += first_name
        if last_name is not None:
            if person != "":
                person += " "
            person += last_name

        if person != "":
            persons += [{
                "entity": "person",
                "value": person,
                "line id": iline
            }]

        return persons

    def find_addresses(self, line, iline, found_so_far):

        print("finding addresses in line [{}]: {}".format(iline, line))
        for entity in found_so_far:
            if(entity["entity"] == "www" or
               entity["entity"] == "phone" or
               entity["entity"] == "e-mail"):
                line = line.replace(entity["value"], "")

        if line.strip() == "":
            return []

        response = parse_address(line)

        if len(response) > 0:
            is_address = False
            city = None
            country = None
            for item in response:
                if "house" in item.keys():
                    continue
                if "house_number" in item.keys():
                    continue
                is_address = True
                if "city" in item.keys():
                    city = item["city"]
                if "country" in item.keys():
                    country = item["country"]
            if is_address:
                result = [{
                    "entity": "address line",
                    "value": line,
                    "line id": iline,
                }]
                if city is not None:
                    result += [{
                        "entity": "city",
                        "value": city,
                        "line id": iline,
                    }]
                if country is not None:
                    result += [{
                        "entity": "country",
                        "value": country,
                        "line id": iline,
                    }]
                return result

        return []

    def find_others(self, line, iline, found_so_far):

        for entity in found_so_far:
            if(entity["entity"] == "www" or
               entity["entity"] == "phone" or
               entity["entity"] == "e-mail"or
               entity["entity"] == "address line"):
                line = ""
            if entity["entity"] == "person":
                words = entity["value"].split()
                for word in words:
                    line = line.replace(word, "")

        if line.strip() == "":
            return []

        tfidf = self.org_occ_model["tfidf"].transform([line])
        pca = self.org_occ_model["pca"].transform(tfidf.toarray())
        qTrf = self.org_occ_model["qTrf"].transform(pca)
        mlp = self.org_occ_model["mlp"].predict(qTrf)
        mlp_prob = self.org_occ_model["mlp"].predict_proba(qTrf)
        prediction = self.org_occ_model["encoder"].inverse_transform(mlp)

        return [{
            "entity": prediction[0],
            "value": line,
            "line id": iline,
            "prob": np.max(mlp_prob),
        }]


if __name__ == "__main__":

    import json

    test_data = [
        [
            'Member of',
            'The University of lowa',
            'Email: zaidan@cem.ch',
            'Phone: (0033)669749919',
            '*',
            'Office: CERN, CH- 1211',
            'Geneva 23- Switzerland',
            'Rémi Zaidan §',
            'Postdoctoral Research Assistant'
        ],
        [
            'Member of',
            'The University of lowa',
            'Email: zaidan@cem.ch',
            'Phone: (0033)669749919',
            '*',
            'Office: CERN, CH- 1211',
            'Geneva 23- Switzerland',
            "R\u00e9mi Zaidan \u00a7",
            'Postdoctoral Research Assistant'
        ],
        [
            "Member of",
            "APS",
            "Physics",
            "www.aps.org",
            "Remi Zaidan",
            "Postdoctoral Research Assistant",
            "Email: zaidan@cern.ch",
            "Phone: (0033) 669749919",
            "The University of Iowa",
            "Office: CERN, CH-1201",
            "Geneva 23, Switzerland",
            "American Physics Society"
        ]
    ]

    ner = NER()

    for lines in test_data:

        print("=====================================================")
        print(json.dumps(lines, indent=2))
        print("-----------------------------------------------------")
        result = ner.parse(lines)
        print(json.dumps(result, indent=2))
        print("=====================================================")
