from setuptools import setup, find_packages

with open("README.md", "r") as fh:
  long_description = fh.read()

with open("requirements.txt") as fh:
  install_requires = fh.read()

setup(
  name="aomenc-by-gop",
  version="0.1.2",
  author="wwwwwwww",
  author_email="wvvwvvvvwvvw@gmail.com",
  description="Aomenc by gop",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/wwww-wwww/py-aomenc-by-gop",
  install_requires=install_requires,
  packages=find_packages(),
  package_data={"aomenc_by_gop": ["bin/linux_amd64/*", "bin/win64/*"]},
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Operating System :: OS Independent",
  ],
  python_requires=">=3.6",
  zip_safe=False,
  entry_points={
    "console_scripts": ["aomenc-by-gop=aomenc_by_gop.app:main"],
  },
)
