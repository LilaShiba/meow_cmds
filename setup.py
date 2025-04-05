from setuptools import setup


setup(
    name='cli-tools',
    version='1.0',
    packages=['.', 'cmds'],
    py_modules=['weather', 'neo', 'wut', 'sky',
                's_array', 'lights'],
    install_requires=[
        'requests',
        'python-dotenv'
    ],
    entry_points={
        'console_scripts': [
            'weather = weather:main',
            'neo = neo:main',
            'wut = define:main',
            'sky = sky:main',
            'weather_log = weather_logger:main',
            'sensors = s_array:main',
            'lights = lights:main'
        ],
    },
)
