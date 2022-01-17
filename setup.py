#!/usr/bin/env python
from setuptools import (
    find_packages,
    setup
)


with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ['PySide6>=6.1.2',
                'pandas==1.3.4',
                'pyqtgraph==0.12.3 ',
                'mne==0.24.1',
                'explorepy',
                'numpy==1.21.4'
                ]

test_requirements = ["pytest==6.2.5",
                     "flake8==4.0.1",
                     "isort==5.10.1",
                     "pytest-qt==4.0.2"]
extras = {"test": test_requirements}

setup(
    author="Mentalab GmbH.",
    author_email='support@mentalab.com',
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Software Development',
    ],
    description="Explore GUI",
    install_requires=requirements,
    long_description=readme + '\n\n',
    include_package_data=True,
    keywords='exploregui',
    name='exploregui',
    packages=find_packages(include=['exploregui', 'exploregui.*']),
    test_suite='tests',
    extras_require=extras,
    url='https://github.com/Mentalab-hub/explorepy-gui',
    version='0.1.0',
    zip_safe=False,
    entry_points={'console_scripts': ['exploregui = exploregui.main:main']},
)
