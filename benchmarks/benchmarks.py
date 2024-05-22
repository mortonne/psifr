from importlib import resources
import pandas as pd
from psifr import fr


class TimeFR:
    """Time common manipulations of free-recall data."""

    def __init__(self, study, engine='python', dtype_backend='numpy_nullable'):
        self.study = study
        self.data_file = str(resources.files('psifr') / 'data' / f'{study}.csv')
        self.raw = None
        self.data = None
        self.category = True if study == 'Morton2013' else False
        self.engine = engine
        self.dtype_backend = dtype_backend

    def setup(self):
        opt = {'engine': self.engine, 'dtype_backend': self.dtype_backend}
        if self.category:
            self.raw = pd.read_csv(
                self.data_file, dtype={'category': 'category'}, **opt
            )
            self.raw.category = self.raw.category.cat.as_ordered()
            self.data = fr.merge_free_recall(
                self.raw,
                list_keys=['list_type', 'list_category'],
                study_keys=['category'],
            )
        else:
            self.raw = pd.read_csv(self.data_file, **opt)
            self.data = fr.merge_free_recall(self.raw)

    def time_read(self):
        raw = pd.read_csv(self.data_file)

    def time_merge(self):
        if self.category:
            data = fr.merge_free_recall(
                self.raw, list_keys=['list_type', 'list_category'], study_keys=['category']
            )
        else:
            data = fr.merge_free_recall(self.raw)

    def time_lag_crp(self):
        crp = fr.lag_crp(self.data)

    def time_lag_crp_compound(self):
        crp = fr.lag_crp_compound(self.data)

    def time_category_crp(self):
        if not self.category:
            raise ValueError(f'Cannot run category crp test on {self.study} dataset.')
        crp = fr.category_crp(self.data, 'category')

    def time_spc(self):
        spc = fr.spc(self.data)
