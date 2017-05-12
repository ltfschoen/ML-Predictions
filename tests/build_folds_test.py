import unittest
import pandas as pd

import sys
import site

def get_main_path():
    test_path = sys.path[0] # sys.path[0] is current path in lib subdirectory
    split_on_char = "/"
    return split_on_char.join(test_path.split(split_on_char)[:-1])
main_path = get_main_path()
site.addsitedir(main_path+'/tests')
site.addsitedir(main_path+'/lib')
print ("Imported subfolder: %s" % (main_path+'/tests') )
print ("Imported subfolder: %s" % (main_path+'/lib') )

from lib.build_folds import build_folds

class BuildFoldsCase(unittest.TestCase):
    """Tests for `build_folds.py`."""

    def setUp(self):
        """"""

    def tearDown(self):
        """"""

    def build_df(self, df_len):
        return pd.DataFrame({ 'A': list(range(1, df_len + 1)) })

    def test_build_folds_manually_generates_lower_upper_extents_for_given_folds(self):
        # Tests
        self.assertEqual(build_folds(self.build_df(7), 8), [0, 1, 2, 3, 4, 5, 6, 7]) # K-Folds 8 error Invalid folds for length
        self.assertEqual(build_folds(self.build_df(7), 7), [0, 1, 2, 3, 4, 5, 6, 7]) # K-Folds 7 error
        self.assertEqual(build_folds(self.build_df(7), 6), [0, 1, 2, 3, 4, 5, 7]) # K-Folds 6 error
        self.assertEqual(build_folds(self.build_df(7), 5), [0, 1, 2, 3, 4, 7]) # K-Folds 5 error
        self.assertEqual(build_folds(self.build_df(7), 4), [0, 2, 4, 6, 7]) # K-Folds 4 error
        self.assertEqual(build_folds(self.build_df(7), 3), [0, 2, 4, 7]) # K-Folds 3 error
        self.assertEqual(build_folds(self.build_df(7), 2), [0, 4, 7]) # K-Folds 2 error
        self.assertEqual(build_folds(self.build_df(7), 1), [0, 7]) # K-Folds 1 error
        self.assertEqual(build_folds(self.build_df(7), 0), [0, 7]) # K-Folds 0 error Invalid folds for length

if __name__ == '__main__':
    unittest.main()
