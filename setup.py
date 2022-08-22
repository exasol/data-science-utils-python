# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['exasol_data_science_utils_python',
 'exasol_data_science_utils_python.model_utils',
 'exasol_data_science_utils_python.model_utils.udfs',
 'exasol_data_science_utils_python.preprocessing',
 'exasol_data_science_utils_python.preprocessing.scikit_learn',
 'exasol_data_science_utils_python.preprocessing.sql',
 'exasol_data_science_utils_python.preprocessing.sql.encoding',
 'exasol_data_science_utils_python.preprocessing.sql.normalization',
 'exasol_data_science_utils_python.preprocessing.sql.schema',
 'exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn',
 'exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.encoding',
 'exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.normalization',
 'exasol_data_science_utils_python.udf_utils',
 'exasol_data_science_utils_python.utils']

package_data = \
{'': ['*']}

install_requires = \
['exasol-bucketfs-utils-python @ '
 'git+https://github.com/exasol/bucketfs-utils-python.git@main',
 'joblib>=1.0.1,<2.0.0',
 'jsonpickle>=2.0.0,<3.0.0',
 'mlxtend>=0.20.0,<0.21.0',
 'pandas>=1.1.0,<2.0.0',
 'pyexasol>=0.25.0,<0.26.0',
 'scikit-learn>=1.1.2,<2.0.0',
 'simplejson>=3.17.2,<4.0.0',
 'tenacity>=8.0.1,<9.0.0',
 'typeguard>=2.11.1,<3.0.0']

setup_kwargs = {
    'name': 'exasol-data-science-utils-python',
    'version': '0.1.0',
    'description': 'Exasol specific Data Science utilities for the Python programming language',
    'long_description': 'This project provides utilities for developing data science integrations for Exasol.\n\n** Note: This project is in a very early development phase. **',
    'author': 'Torsten Kilias',
    'author_email': 'torsten.kilias@exasol.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/exasol/data-science-utils-python',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<4.0',
}


setup(**setup_kwargs)
