# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['exasol_data_science_utils_python',
 'exasol_data_science_utils_python.model_utils',
 'exasol_data_science_utils_python.preprocessing',
 'exasol_data_science_utils_python.preprocessing.encoding',
 'exasol_data_science_utils_python.preprocessing.normalization',
 'exasol_data_science_utils_python.preprocessing.schema',
 'exasol_data_science_utils_python.udf_utils']

package_data = \
{'': ['*']}

install_requires = \
['exasol-bucketfs-utils-python @ '
 'git+https://github.com/exasol/bucketfs-utils-python.git@main',
 'joblib>=1.0.1,<2.0.0',
 'jsonpickle>=2.0.0,<3.0.0',
 'mlxtend>=0.18.0,<0.19.0',
 'pandas>=1.1.0,<2.0.0',
 'pyexasol>=0.17.0,<0.18.0',
 'scikit-learn>=0.24.1,<0.25.0',
 'simplejson>=3.17.2,<4.0.0']

setup_kwargs = {
    'name': 'exasol-data-science-utils-python',
    'version': '0.1.0',
    'description': 'Exasol specific Data Science utilities for the Python programming language',
    'long_description': '',
    'author': 'Torsten Kilias',
    'author_email': 'torsten.kilias@exasol.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/exasol/data-science-utils-python',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.6.1,<4.0',
}


setup(**setup_kwargs)
