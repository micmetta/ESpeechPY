#!/Users/leisen/anaconda3/bin/python3
# -*- coding: UTF-8 -*-

# ********************************************************
# * Author        : LEI Sen
# * Email         : sen.lei@outlook.com
# * Create time   : 2018-11-30 13:30
# * Last modified : 2018-11-30 13:30
# * Filename      : main.py
# * Description   : 
# *********************************************************

import sys
import os
import click
import subprocess
import shlex
import platform

HERE = os.path.dirname(os.path.abspath(__file__))
OS_TYPE = platform.platform().split('-')[0]

@click.command()
@click.option(
    'pkg_name', '--name', '-n',
    help="Specify the package name. ",
    required=False,
#    prompt="Please specify package name",
)
@click.option(
    'pkg_version', '--version', '-v',
    help="Specify the package version number. ",
    required=False,
#    prompt="Please specify (new) package version number",
)
@click.option(
    'real_pypi', '--real', '-r',
    default=False,
    is_flag=True,
    help="Use the real PyPi index (instead of test PyPi by default). ",
    required=False,
#    prompt='Are you sure you want to use real PyPi index (instead of test PyPi)? '
)
#@click.argument('pkg_dir', nargs=1, type=click.STRING, required=True)
@click.argument(
    'pkg_dir', 
    nargs=1, 
    type=click.Path(exists=True, file_okay=False, writable=True), 
    required=True, 
#    default='.',
)
def main(pkg_dir, pkg_name, pkg_version, real_pypi):
    """
    Python command line tool to setup Python package automatically. 
    \b
    Example:
    \b
    \t $ autopypi your-package-root-directory -n package_name -v package_version -r
    \t 
    """
    if real_pypi:
        click.echo("")
        #click.echo("! Using REAL PyPi index ! ")
        if OS_TYPE=='Darwin':
            text_show = """echo "! Using \x1b[32;1mREAL \x1b[0mPyPi index ! " """
            text_show_r = """echo "Uploading to \x1b[32;1mReal \x1b[0mPyPi index ... " """
        elif OS_TYPE=='Linux':
            text_show = """echo "! Using \e[32m\e[5mREAL \e[39m\e[25mPyPi index ! " """
            text_show_r = """echo "Uploading to \e[32m\e[5mReal \e[39m\e[25mPyPi index ... " """
        elif os_type == 'Windows':
            text_show = """echo "! Using REAL PyPi index ! " """
            text_show_r = """echo "Uploading to Real PyPi index ... " """
        os.system(text_show)

    else:
        click.echo("")
        #click.echo("! Using TEST PyPi index ! ")
        if OS_TYPE=='Darwin':
            text_show = """echo "! Using \x1b[32;1mTEST \x1b[0mPyPi index ! " """
            text_show_t = """echo "Uploading to \x1b[32;1mTest \x1b[0mPyPi index ... " """
        elif OS_TYPE=='Linux':
            text_show = """echo "! Using \e[32m\e[5mTEST \e[39m\e[25mPyPi index ! " """
            text_show_t = """echo "Uploading to \e[32m\e[5mTest \e[39m\e[25mPyPi index ... " """
        elif os_type == 'Windows':
            text_show = """echo "! Using TEST PyPi index ! " """
            text_show_t = """echo "Uploading to Test PyPi index ... " """

        os.system(text_show)

    click.echo("  Setting up package: [{}]-v{} ".format(pkg_name, pkg_version))
    click.echo("")

    ## Remove old folders for uploading
    command_script_rm = """
if [ -d $dist_folder ]; then
    echo "  removing old dist folder ... "
    rm -r """ + pkg_dir + "/dist/" + """
else
    echo "  dist folder not found. "
fi

if [ -d $build_folder ]; then
    echo "  removing old build folder ... "
    rm -r """ + pkg_dir + "/build/" + """
else
    echo "  build folder not found. "
fi
    """
    os.system(command_script_rm)

    os.system("python3 -m pip install --user --upgrade setuptools wheel")
    os.system("python3 -m pip install --user --upgrade twine")

    ## Check if setup.py file exists
    setup_file = pkg_dir + "/setup.py"
    is_file = os.path.isfile(setup_file)
    if not is_file:
        print("\n")
        #raise FileNotFoundError("<setup.py> file NOT found. Aborted !")
        print("\u001b[35m<setup.py> file NOT found. Aborted !")
        #os.system("exit")
        sys.exit()

    print("\n")
    os.system("python3 setup.py sdist bdist_wheel")
    print("\nChecking on the sdist and wheel ... (also check README rendering problem)")
    os.system("twine check " + pkg_dir + "/dist/*")
    print("\n")

    if real_pypi:
        #twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
        #print("Uploading to Real PyPi index ... ")
        os.system(text_show_r)
        command_script = 'twine upload --repository-url https://upload.pypi.org/legacy/' + ' ' + pkg_dir + '/dist/*'
        os.system(command_script)
    else:
        #twine upload --repository-url https://test.pypi.org/legacy/ dist/*
        #subprocess.Popen(["bash", "./setup_new_pypi.sh"])
        #print("Uploading to Test PyPi index ... ")
        os.system(text_show_t)
        command_script = 'twine upload --repository-url https://test.pypi.org/legacy/' + ' ' + pkg_dir + '/dist/*'
        os.system(command_script)
