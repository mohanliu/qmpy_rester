import json
import requests
import os

Rester_ENDPOINT = os.environ.get("oqmd_url", "http://oqmd.org")


class QMPYRester(object):
    def __init__(self, endpoint=Rester_ENDPOINT):
        self.preamble = endpoint
        self.session = requests.Session()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()

    def _make_requests(self, sub_url, payload=None, method="GET"):
        response = None
        url = self.preamble + sub_url

        if method == "GET":
            response = self.session.get(url, params=payload, verify=True)

            if response.status_code in [200, 400]:
                data = json.loads(response.text)
                return data

    def get_oqmd_phases(self, verbose=True, **kwargs):
        """
        Input:
            verbose: boolean
            **kwargs: dict
        Output:
            dict
        """

        # URL paramters
        url_args = []
        kwargs_list = [
            "composition",
            "icsd",
            "filter",
            "sort_by",
            "desc",
            "sort_offset",
            "limit",
            "offset",
        ]

        # Attributes for filters
        filter_args = []
        filter_list = [
            "element_set",
            "element",
            "spacegroup",
            "prototype",
            "generic",
            "volume",
            "natoms",
            "ntypes",
            "stability",
            "delta_e",
            "band_gap",
        ]

        for k in kwargs.keys():
            if k in kwargs_list:
                url_args.append("%s=%s" % (k, kwargs[k]))
            elif k in filter_list:
                if ">" in kwargs[k] or "<" in kwargs[k]:
                    filter_args.append("%s%s" % (k, kwargs[k]))
                else:
                    filter_args.append("%s=%s" % (k, kwargs[k]))
            elif k == "fields":
                if "!" in kwargs[k]:
                    url_args.append("fields!=%s" % kwargs[k].replace("!", ""))
                else:
                    url_args.append("fields=%s" % kwargs[k])

        if filter_args != []:
            filters_tag = " AND ".join(filter_args)
            url_args.append("filter=" + filters_tag)

        if verbose:
            print("Your filters are:")
            if url_args == []:
                print("   No filters?")
            else:
                for arg in url_args:
                    print("   ", arg)

            ans = input("Proceed? [Y/n]:")

            if ans not in ["Y", "y", "Yes", "yes"]:
                return

        _url = "&".join(url_args)
        self.suburl = _url

        return self._make_requests("/oqmdapi/formationenergy?%s" % _url)

    def get_oqmd_phase_space(self, space, **kwargs):
        """
        Input:
            space: str
                e.g. 'Al-O', 'Pd-Se-Te-Pb'
            verbose: boolean
            **kwargs: dict
        Output:
            dict
        """
        _url = "composition={}".format(space)
        _url += "&fields=name,delta_e,stability"
        _url += "&limit=200"
        output_data = self._make_requests("/oqmdapi/formationenergy?%s" % _url)

        while output_data["links"]["next"]:
            new_data = self._make_requests(
                output_data["links"]["next"].replace(self.preamble, "")
            )
            output_data["data"].extend(new_data["data"])
            output_data["links"]["next"] = new_data["links"]["next"]
            output_data["meta"]["data_returned"] += new_data["meta"]["data_returned"]

        output_data["meta"]["more_data_available"] = False

        return output_data

    def get_oqmd_phase_by_id(self, fe_id, fields=None):
        if fields:
            if "!" in fields:
                ex_fields = fields.replace("!", "")
                return self._make_requests(
                    "/oqmdapi/formationenergy/%d?fields!=%s" % (fe_id, ex_fields)
                )
            else:
                return self._make_requests(
                    "/oqmdapi/formationenergy/%d?fields=%s" % (fe_id, fields)
                )

        return self._make_requests("/oqmdapi/formationenergy/%d" % fe_id)

    def get_optimade_structures(self, verbose=True, **kwargs):
        """
        Input:
            verbose: boolean
            **kwargs: dict
        Output:
            dict
        """

        # URL paramters
        url_args = []
        kwargs_list = ["limit", "offset", "filter"]

        # Attributes for filters
        filter_args = []
        filter_list = [
            "elements",
            "nelements",
            "chemical_formula",
            "formula_prototype",
            "_oqmd_volume",
            "_oqmd_spacegroup",
            "_oqmd_natoms",
            "_oqmd_prototype",
            "_oqmd_stability",
            "_oqmd_delta_e",
            "_oqmd_band_gap",
        ]

        for k in kwargs.keys():
            if k in kwargs_list:
                url_args.append("%s=%s" % (k, kwargs[k]))
            elif k in filter_list:
                if ">" in kwargs[k] or "<" in kwargs[k]:
                    filter_args.append("%s%s" % (k, kwargs[k]))
                else:
                    filter_args.append("%s=%s" % (k, kwargs[k]))
            elif k == "fields":
                if "!" in kwargs[k]:
                    url_args.append("fields!=%s" % kwargs[k].replace("!", ""))
                else:
                    url_args.append("fields=%s" % kwargs[k])

        if filter_args != []:
            filters_tag = " AND ".join(filter_args)
            url_args.append("filter=" + filters_tag)

        if verbose:
            print("Your filters are:")
            if url_args == []:
                print("   No filters?")
            else:
                for arg in url_args:
                    print("   ", arg)

            ans = input("Proceed? [Y/n]:")

            if ans not in ["Y", "y", "Yes", "yes"]:
                return

        _url = "&".join(url_args)
        self.suburl = _url

        return self._make_requests("/optimade/structures?%s" % _url)

    def get_optimade_structure_by_id(self, id, fields=None):
        if fields:
            if "!" in fields:
                ex_fields = fields.replace("!", "")
                return self._make_requests(
                    "/optimade/structures/%d?fields!=%s" % (id, ex_fields)
                )
            else:
                return self._make_requests(
                    "/optimade/structures/%d?fields=%s" % (id, fields)
                )

        return self._make_requests("/optimade/structures/%d" % id)
