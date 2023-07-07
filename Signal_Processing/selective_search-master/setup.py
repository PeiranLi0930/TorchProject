from setuptools import setup, find_packages

with open('README.md', encoding='utf8') as f:
    readme = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='selective_search',
    version='1.1.0',
    url='https://github.com/ChenjieXu/selective_search',
    description='Selective Search in Python',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Chenjie Xu',
    author_email='cxuscience@gmail.com',
    keywords='selective_search',
    packages=find_packages(),
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires=reqs.strip().split('\n'),
)
