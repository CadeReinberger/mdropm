#!/usr/bin/env python3
"""Compatibility entry point for the reproducible canting analysis."""

# This file formerly hard-coded four selected results and fit parameters.  Keep
# the convenient command name, but always analyze the saved trajectories.
from canting_plotter import main


if __name__ == "__main__":
    main()
