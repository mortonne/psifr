from pkg_resources import resource_filename
import pandas as pd
from psifr import fr


class TimeFR:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    def setup(self):
        data_file = resource_filename('psifr', 'data/Morton2013.csv')
        self.raw = pd.read_csv(data_file, dtype={'category': 'category'})
        self.raw.category.cat.as_ordered(inplace=True)
        self.data = fr.merge_free_recall(
            self.raw, list_keys=['list_type', 'list_category'], study_keys=['category']
        )

    def time_merge(self):
        data = fr.merge_free_recall(
            self.raw, list_keys=['list_type', 'list_category'], study_keys=['category']
        )

    def time_lag_crp(self):
        crp = fr.lag_crp(self.data)

    def time_category_crp(self):
        crp = fr.category_crp(self.data, 'category')

    def time_spc(self):
        spc = fr.spc(self.data)
