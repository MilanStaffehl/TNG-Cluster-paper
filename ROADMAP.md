# Roadmap

A brief overview over the steps taken. The numbers match the subdirectory names in the directories `scripts/`, `figures/` and `data/`. 


## 001: Temperature distributions

**Task:** Plot the distribution of gas temperature in different mass bins.

**Goal:** Determine what temperature can be used to separate cool from hot gas or cool from warm and warm from hot gas.

**Result:** Using gas mass fraction for all subsequent tasks is better suited to not lose those halos that have low total gas mass but still might be interesing. The temperature divisions between cool, warm and hot gas are chosen in two ways: once as absolute values, once as fractions of the virial temperature.

GitHub issues for tasks:

- [x] [#4 Plot: plot temperature distribution in halos](https://github.com/MilanStaffehl/thesisProject/issues/4)
- [x] [#11 Plot: distribution of temperatures weighted by mass](https://github.com/MilanStaffehl/thesisProject/issues/11)
- [x] [#13 Plot: treat star-forming gas differently in plot](https://github.com/MilanStaffehl/thesisProject/issues/13)
- [x] [#23 Explore variations: plot gallery](https://github.com/MilanStaffehl/thesisProject/issues/23)
- [x] [#27 Plot temperature distribution normalized to virial temperature](https://github.com/MilanStaffehl/thesisProject/issues/27)

Typical directory name for related code: `temperature_distribution`, `temperature_histograms`
 

## 002: Radial temperature profiles

**Task:** Plot a 2D histogram of temperature-radial distance weighted by gas fraction.

**Goal:** Determine from this the radial temperature profile and identify regions which hold cool gas.

**Result:** pending

GitHub issues for tasks:

- [ ] [#26 Plot temperature distribution normalized to virial temperature](https://github.com/MilanStaffehl/thesisProject/issues/26)

Typical directory name for related code: `radial_profiles`


## 003: Gas fraction trends with halo mass

**Task:** Plot the trend of cool, warm and hot gas mass (fraction) with halo mass.

**Goal:** Determine whether the cut-off temperature needs to vary with mass or if it is constant across halo masses.

**Result:** pending

GitHub issues for tasks:

- [ ] [#12 Plot: determine gas mass vs. halo mass for all regimes](https://github.com/MilanStaffehl/thesisProject/issues/12)

Typical directory name for related code: `mass_trends`
