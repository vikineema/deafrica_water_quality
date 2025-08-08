from importlib.metadata import PackageNotFoundError, version

import click


@click.command()
def cli():
    """Show the version from pyproject.toml."""
    package_name = "deafrica-water-quality"
    try:
        click.echo(version(package_name))
    except PackageNotFoundError:
        click.echo("Package not installed", err=True)
