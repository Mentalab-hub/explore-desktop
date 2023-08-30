#!/usr/bin/env python
from setuptools import (
    find_packages,
    setup
)


with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ['PySide6==6.4.1',  # working v: 6.2.1 (not in p3.11)
                'pandas==1.3.4',  # 2.0.3
                'pyqtgraph==0.12.3',  # 0.13.3
                'mne==0.24.1',  # 1.5.0
                'explorepy',
                'scipy==1.11.2',  # working v: 1.7.3 (not in p3.11), 1.9.0 fails installing?
                'numpy==1.25.2'  # working: v: 1.21.4 (not in p3.11), 1.22.0 clashes with eeglabio? on p3.11??????????????
                ]

test_requirements = ["pytest",
                     "flake8",
                     "isort",
                     "pytest-qt"]
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
    description="Explore Desktop",
    install_requires=requirements,
    long_description=readme + '\n\n',
    include_package_data=True,
    keywords='exploredesktop',
    name='exploredesktop',
    packages=find_packages(include=['exploredesktop', 'exploredesktop.*']),
    test_suite='tests',
    extras_require=extras,
    url='https://github.com/Mentalab-hub/explore-desktop',
    version='0.7.1',
    zip_safe=False,
    entry_points={'console_scripts': ['exploredesktop = exploredesktop.main:main']},
)
