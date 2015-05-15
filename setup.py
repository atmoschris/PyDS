from setuptools import setup

setup(
    name = 'pyds',
    version=0.1,
    description = 'PyDS: Dynamical Systems Running with Python',
    long_description=open('README.rst').read(),
    author='Feng Zhu',
    author_email='feng.zhu@wisc.edu',
    url = 'https://github.com/lyricorpse/PyDS',
    license='BSD',
    py_modules = ['pyds'],
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Programming Language :: Python :: 3'
    ]
)

