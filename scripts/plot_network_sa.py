# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

"""
Plots map with pie charts and cost box bar charts.
Relevant Settings
-----------------
Inputs
------
Outputs
-------
Description
-----------
"""

import logging

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from _helpers import (
    aggregate_costs,
    aggregate_p,
    configure_logging,
    load_network_for_plots,
)
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, Ellipse

<<<<<<< HEAD
from add_electricity import add_emission_prices
import copy

=======
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
to_rgba = mpl.colors.colorConverter.to_rgba

logger = logging.getLogger(__name__)


def make_handler_map_to_scale_circles_as_in(ax, dont_resize_actively=False):
    fig = ax.get_figure()

    def axes2pt():
        return np.diff(ax.transData.transform([(0, 0), (1, 1)]), axis=0)[0] * (
<<<<<<< HEAD
                72.0 / fig.dpi
=======
            72.0 / fig.dpi
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
        )

    ellipses = []
    if not dont_resize_actively:

        def update_width_height(event):
            dist = axes2pt()
            for e, radius in ellipses:
                e.width, e.height = 2.0 * radius * dist

        fig.canvas.mpl_connect("resize_event", update_width_height)
        ax.callbacks.connect("xlim_changed", update_width_height)
        ax.callbacks.connect("ylim_changed", update_width_height)

    def legend_circle_handler(
<<<<<<< HEAD
            legend, orig_handle, xdescent, ydescent, width, height, fontsize
=======
        legend, orig_handle, xdescent, ydescent, width, height, fontsize
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
    ):
        w, h = 2.0 * orig_handle.get_radius() * axes2pt()
        e = Ellipse(
            xy=(0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent),
            width=w,
            height=w,
        )
        ellipses.append((e, orig_handle.get_radius()))
        return e

    return {Circle: HandlerPatch(patch_func=legend_circle_handler)}


def make_legend_circles_for(sizes, scale=1.0, **kw):
    return [Circle((0, 0), radius=(s / scale) ** 0.5, **kw) for s in sizes]


def set_plot_style():
    plt.style.use(
        [
            "classic",
            "seaborn-v0_8-whitegrid",
<<<<<<< HEAD
            # "seaborn-white",
=======
            #"seaborn-white",
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
            {
                "axes.grid": False,
                "grid.linestyle": "--",
                "grid.color": "0.6",
                "hatch.color": "white",
                "patch.linewidth": 0.5,
                "font.size": 12,
                "legend.fontsize": "medium",
                "lines.linewidth": 1.5,
                "pdf.fonttype": 42,
            },
        ]
    )


def plot_map(n, opts, ax=None, attribute="p_nom"):
    if ax is None:
        ax = plt.gca()

    ## DATA
    line_colors = {
        "cur": "purple",
        "exp": mpl.colors.rgb2hex(to_rgba("red", 0.7), True),
    }
    tech_colors = opts["tech_colors"]

    if attribute == "p_nom":
        # bus_sizes = n.generators_t.p.sum().loc[n.generators.carrier == "load"].groupby(n.generators.bus).sum()
        n.generators.loc[n.generators.carrier.isin(["OCGT", "CCGT"]), "carrier"] = "gas"
<<<<<<< HEAD
        n.generators.loc[n.generators.carrier.isin(["hydro", "hydro-import", "hydro+PHS"]), "carrier"] = "hydro"
=======
        n.generators.loc[n.generators.carrier.isin(["hydro", "hydro-import"]), "carrier"] = "hydro"
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
        bus_sizes = pd.concat(
            (
                n.generators.query('carrier != "load_shedding"')
                .groupby(["bus", "carrier"])
                .p_nom_opt.sum(),
                n.storage_units.groupby(["bus", "carrier"]).p_nom_opt.sum(),
            )
        )
        line_widths_exp = n.lines.s_nom_opt
        line_widths_cur = n.lines.s_nom_min
        link_widths_exp = n.links.p_nom_opt
        link_widths_cur = n.links.p_nom_min
    else:
        raise "plotting of {} has not been implemented yet".format(attribute)

    line_colors_with_alpha = (line_widths_cur / n.lines.s_nom > 1e-3).map(
        {True: line_colors["cur"], False: to_rgba(line_colors["cur"], 0.0)}
    )
    link_colors_with_alpha = (link_widths_cur / n.links.p_nom > 1e-3).map(
        {True: line_colors["cur"], False: to_rgba(line_colors["cur"], 0.0)}
    )

    ## FORMAT
    linewidth_factor = opts["map"][attribute]["linewidth_factor"]
    bus_size_factor = opts["map"][attribute]["bus_size_factor"]

    supply_regions = gpd.read_file('/home/agatha/Desktop/pypsa-za-master/data/bundle/supply_regions/27-supply.shp')
<<<<<<< HEAD
    # supply_regions = gpd.read_file(snakemake.input.supply_regions)
=======
    #supply_regions = gpd.read_file(snakemake.input.supply_regions)
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
    resarea = gpd.read_file(snakemake.input.resarea)

    supply_regions.plot(ax=ax, facecolor='none', edgecolor='black')
    resarea.plot(ax=ax, facecolor='gray', alpha=0.2)

    ## PLOT
    n.plot(
        line_widths=line_widths_exp / linewidth_factor,
        link_widths=link_widths_exp / linewidth_factor,
        line_colors=line_colors["cur"],
        link_colors=line_colors["cur"],
        bus_sizes=bus_sizes / bus_size_factor,
        bus_colors=tech_colors,
        boundaries=map_boundaries,
        color_geomap=True,
        geomap=True,
        ax=ax,
    )
    n.plot(
        line_widths=line_widths_cur / linewidth_factor,
        link_widths=link_widths_cur / linewidth_factor,
        line_colors=line_colors_with_alpha,
        link_colors=link_colors_with_alpha,
        bus_sizes=0,
        boundaries=map_boundaries,
        color_geomap=True,
        geomap=True,
        ax=ax,
    )
    ax.set_aspect("equal")
    ax.axis("off")

    # Rasterize basemap
    # TODO : Check if this also works with cartopy
    for c in ax.collections[:2]:
        c.set_rasterized(True)

    # LEGEND
    handles = []
    labels = []

    for s in (10, 1):
        handles.append(
            plt.Line2D(
                [0], [0], color=line_colors["cur"], linewidth=s * 1e3 / linewidth_factor
            )
        )
        labels.append("{} GW".format(s))
    l1_1 = ax.legend(
        handles,
        labels,
        loc="upper left",
<<<<<<< HEAD
        bbox_to_anchor=(0.3, 1.01),
=======
        bbox_to_anchor=(0.24, 1.01),
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
        frameon=False,
        labelspacing=0.8,
        handletextpad=1.5,
        title="Transmission",
    )
    ax.add_artist(l1_1)

    handles = []
    labels = []
    for s in (10, 5):
        handles.append(
            plt.Line2D(
                [0], [0], color=line_colors["cur"], linewidth=s * 1e3 / linewidth_factor
            )
        )
<<<<<<< HEAD
        # labels.append("/")
    #    l1_2 = ax.legend(
    #        handles,
    #        labels,
    #        loc="upper left",
    #        bbox_to_anchor=(0.26, 1.01),
    #        frameon=False,
    #        labelspacing=0.8,
    #        handletextpad=0.5,
    #        title=" ",
    #    )
    #    ax.add_artist(l1_2)
=======
        #labels.append("/")
#    l1_2 = ax.legend(
#        handles,
#        labels,
#        loc="upper left",
#        bbox_to_anchor=(0.26, 1.01),
#        frameon=False,
#        labelspacing=0.8,
#        handletextpad=0.5,
#        title=" ",
#    )
#    ax.add_artist(l1_2)
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173

    handles = make_legend_circles_for(
        [10e3, 5e3, 1e3], scale=bus_size_factor, facecolor="w"
    )
    labels = ["{} GW".format(s) for s in (10, 5, 3)]
    l2 = ax.legend(
        handles,
        labels,
        loc="upper left",
<<<<<<< HEAD
        bbox_to_anchor=(0.02, 1.01),
=======
        bbox_to_anchor=(0.01, 1.01),
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
        frameon=False,
        labelspacing=1.0,
        title="Capacity",
        handler_map=make_handler_map_to_scale_circles_as_in(ax),
    )
    ax.add_artist(l2)

    techs = (bus_sizes.index.levels[1]).intersection(
        pd.Index(opts["vre_techs"] + opts["conv_techs"] + opts["storage_techs"])
    )
<<<<<<< HEAD

    custom_order = ['Coal', 'Nuclear', 'Gas', 'Wind', 'CSP', 'PV', 'Hydro', 'Battery']
    handles = []
    labels = []

    for t in techs:
        label = opts["nice_names"].get(t, t)

        if label == "Hydro+PS":
            continue  # Skip the rest of this iteration

        handles.append(
            plt.Line2D(
                [0], [0], color=tech_colors[t], marker="o", markersize=14, linewidth=0
            )
        )
        labels.append(label)

    ordered_handles = []
    ordered_labels = []

    # Order according to custom_order
    for lbl in custom_order:
        if lbl in labels:
            idx = labels.index(lbl)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])

    l3 = ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        handletextpad=1.0,
        columnspacing=3,
        handlelength=1.5,
        ncol=4,
        labelspacing=1.0,
        # title="Technology",
=======
    handles = []
    labels = []
    for t in techs:
        handles.append(
            plt.Line2D(
                [0], [0], color=tech_colors[t], marker="o", markersize=8, linewidth=0
            )
        )
        labels.append(opts["nice_names"].get(t, t))
    l3 = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.0),  # bbox_to_anchor=(0.72, -0.05),
        handletextpad=0.0,
        columnspacing=0.5,
        ncol=4,
        title="Technology",
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
    )

    return fig


def plot_total_energy_pie(n, opts, ax=None):
    if ax is None:
        ax = plt.gca()

<<<<<<< HEAD
    ax.set_title("Total Generation \nper Technology", fontdict=dict(fontsize="medium"))
=======
    ax.set_title("Generation per technology", fontdict=dict(fontsize="medium"))
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173

    e_primary = aggregate_p(n).drop("load", errors="ignore").loc[lambda s: s > 0]

    patches, texts, autotexts = ax.pie(
        e_primary,
<<<<<<< HEAD
        startangle=110,
        # labels=e_primary.rename(opts["nice_names"]).index,
=======
        startangle=90,
        labels=e_primary.rename(opts["nice_names"]).index,
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
        autopct="%.0f%%",
        shadow=False,
        colors=[opts["tech_colors"][tech] for tech in e_primary.index],
    )
<<<<<<< HEAD
    for autotext in autotexts:
        x, y = autotext.get_position()
        distance_from_center = (x ** 2 + y ** 2) ** 0.5  # Calculate distance from center
        new_x = x / distance_from_center * 1.15  # Increase distance by 15% as an example
        new_y = y / distance_from_center * 1.15  # Increase distance by 15% as an example
        autotext.set_position((new_x, new_y))
        autotext.set_fontsize(9)  # Adjust the size as needed

=======
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
    for t1, t2, i in zip(texts, autotexts, e_primary.index):
        if e_primary.at[i] < 0.04 * e_primary.sum():
            t1.remove()
            t2.remove()


<<<<<<< HEAD
#############

=======
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
def plot_total_cost_bar(n, opts, ax=None):
    if ax is None:
        ax = plt.gca()

    total_load = (n.snapshot_weightings.generators * n.loads_t.p.sum(axis=1)).sum()
    tech_colors = opts["tech_colors"]

    def split_costs(n):
        costs = aggregate_costs(n).reset_index(level=0, drop=True)
        costs_ex = aggregate_costs(n, existing_only=True).reset_index(
            level=0, drop=True
        )
        return (
            costs["capital"].add(costs["marginal"], fill_value=0.0),
            costs_ex["capital"],
            costs["capital"] - costs_ex["capital"],
            costs["marginal"],
        )

    costs, costs_cap_ex, costs_cap_new, costs_marg = split_costs(n)

<<<<<<< HEAD
    # For Third Bar: Create a deep copy of n and adjust it for emission prices.
    n_copy = copy.deepcopy(n)
    emission_prices = snakemake.config['costs']['emission_prices']
    ep = (pd.Series(emission_prices).rename(lambda x: x + '_emissions') * n_copy.carriers).sum(axis=1)

    # Adjusting the marginal costs
    gen_ep = n_copy.generators.carrier.map(ep) / n_copy.generators.efficiency
    su_ep = n_copy.storage_units.carrier.map(ep) / n_copy.storage_units.efficiency_dispatch

    if 'Ep' not in scenario_opts:
        # If 'Ep' is NOT in scenario_opts, add the emission prices to the marginal costs
        n_copy.generators['marginal_cost'] += gen_ep.fillna(0)
        n_copy.storage_units['marginal_cost'] += su_ep.fillna(0)
    else:
        # If 'Ep' IS in scenario_opts, subtract the emission prices from the marginal costs
        n_copy.generators['marginal_cost'] -= gen_ep.fillna(0)
        n_copy.storage_units['marginal_cost'] -= su_ep.fillna(0)

    # Extract costs for third bar using the adjusted copy
    costs_ep, costs_cap_ex_ep, costs_cap_new_ep, costs_marg_ep = split_costs(n_copy)

    print(costs_marg)
    print(costs_marg_ep)

    costs_graph1 = pd.DataFrame(
=======
    costs_graph2 = pd.DataFrame(
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
        dict(a=costs.drop("load", errors="ignore")),
        index=[
            "AC-AC",
            "AC line",
        ],
    ).dropna()
    bottom = np.array([0.0, 0.0])
    texts = []

<<<<<<< HEAD
    for i, ind in enumerate(costs_graph1.index):
        data = np.asarray(costs_graph1.loc[ind]) / 1e9 / 17.673  # USD #/ total_load
=======
    for i, ind in enumerate(costs_graph2.index):
        data = np.asarray(costs_graph2.loc[ind]) /1e9 /17.673 #USD #/ total_load
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
        ax.bar([0.5], data, bottom=bottom, color=tech_colors[ind], width=0.7, zorder=-1)
        bottom_sub = bottom
        bottom = bottom + data

        if ind in opts["conv_techs"] + ["AC line"]:
            for c in [costs_cap_ex, costs_marg]:
                if ind in c:
<<<<<<< HEAD
                    data_ac = np.asarray([c.loc[ind]]) / 1e9 / 17.673  # USD#/ total_load
                    ax.bar(
                        [0.5],
                        data_ac,
                        linewidth=0.1,
                        edgecolor='black',
                        bottom=bottom_sub,
                        color='darkkhaki',
=======
                    data_ac = np.asarray([c.loc[ind]]) /1e9 /17.673 #USD#/ total_load
                    ax.bar(
                        [0.5],
                        data_ac,
                        linewidth=0,
                        bottom=bottom_sub,
                        color='gray',
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
                        width=0.7,
                        zorder=-1,
                        alpha=0.8,
                    )
                    bottom_sub += data_ac

        if abs(data[-1]) < 5:
            continue

<<<<<<< HEAD
        # text = ax.text(
        #    0.5, (bottom + 0.08 * data)[-1] - 3, opts["nice_names"].get(ind, ind), ha='center'
        # )
        # texts.append(text)

    costs_graph2 = pd.DataFrame(
        dict(a=costs_ep.drop("load", errors="ignore")),
=======
        text = ax.text(
            0.5, (bottom + 0.08 * data)[-1] - 3, opts["nice_names"].get(ind, ind), ha='center'
        )
        texts.append(text)


    costs_graph = pd.DataFrame(
        dict(a=costs.drop("load", errors="ignore")),
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
        index=[
            "coal",
            "nuclear",
            "OCGT",
            "CCGT",
            "gas",
<<<<<<< HEAD
            # "AC-AC",
            # "AC line",
=======
            #"AC-AC",
            #"AC line",
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
            "onwind",
            "CSP",
            "solar",
            "hydro",
            "hydro-import",
            "battery",
<<<<<<< HEAD
            # "H2",
=======
            #"H2",
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
        ],
    ).dropna()
    bottom = np.array([0.0, 0.0])
    texts = []

<<<<<<< HEAD
    for i, ind in enumerate(costs_graph2.index):
        data = np.asarray(costs_graph2.loc[ind]) / 1e9 / 17.673  # USD 2022 #/ total_load
        ax.bar([1.5], data, bottom=bottom, color=tech_colors[ind], width=0.7,
               zorder=-1)  # change for scenario 3 in from 1.5 to 2.5
=======

    for i, ind in enumerate(costs_graph.index):
        data = np.asarray(costs_graph.loc[ind]) /1e9 /17.673 #USD 2022 #/ total_load
        ax.bar([1.5], data, bottom=bottom, color=tech_colors[ind], width=0.7, zorder=-1)
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
        bottom_sub = bottom
        bottom = bottom + data

        if ind in opts["conv_techs"] + ["AC line"]:
<<<<<<< HEAD
            for c in [costs_cap_ex_ep, costs_marg_ep]:
                if ind in c:
                    data_sub = np.asarray([c.loc[ind]]) / 1e9 / 17.673  # USD #/ total_load
=======
            for c in [costs_cap_ex, costs_marg]:
                if ind in c:
                    data_sub = np.asarray([c.loc[ind]]) /1e9 /17.673 #USD #/ total_load
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
                    ax.bar(
                        [1.5],
                        data_sub,
                        linewidth=0,
<<<<<<< HEAD
                        edgecolor='black',
=======
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
                        bottom=bottom_sub,
                        color=tech_colors[ind],
                        width=0.7,
                        zorder=-1,
                        alpha=0.8,
                    )
                    bottom_sub += data_sub

<<<<<<< HEAD
=======

>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
        if abs(data[-1]) < 0.25:
            continue

        # Adjust the y position based on half the height of the data bar
        y_position = (bottom - 0.5 * data)[-1]

        # Check if it's the first carrier and adjust the y position accordingly
        if i == 0:
            y_position = (0.5 * data)[-1]

<<<<<<< HEAD
    ### legend
    # text = ax.text(
    #    3.2, y_position, opts["nice_names"].get(ind, ind)
    # )
    # texts.append(text)

    #################

    # Third Bar: With Emissions Prices
    # apply the emission costs to the marginal costs

    # Create a dataframe for plotting, ignoring the 'load' costs.
    costs_graph3 = pd.DataFrame(
        dict(a=costs.drop("load", errors="ignore")), #delete ep in third scenario
        index=[
            "coal", "nuclear", "OCGT", "CCGT", "gas", "onwind", "CSP", "solar", "hydro", "hydro-import", "battery"
        ],
    ).dropna()

    bottom = np.array([0.0, 0.0])
    texts = []

    # Loop through each technology
    for i, ind in enumerate(costs_graph3.index):
        data = np.asarray(costs_graph3.loc[ind]) / 1e9 / 17.673
        ax.bar([2.5], data, bottom=bottom, color=tech_colors[ind], width=0.7, zorder=-1)
        bottom_sub = bottom
        bottom = bottom + data

        # Plot additional cost components for some technologies
        if ind in opts["conv_techs"] + ["AC line"]:
            for c in [costs_cap_ex, costs_marg]: #delete ep in third scenario
                if ind in c:
                    data_sub = np.asarray([c.loc[ind]]) / 1e9 / 17.673
                    ax.bar(
                        [2.5],
                        data_sub,
                        linewidth=0,
                        edgecolor='black',
                        bottom=bottom_sub,
                        color=tech_colors[ind],
                        width=0.7,
                        zorder=-1,
                        alpha=0.8,
                    )
                    bottom_sub += data_sub

        # Skip very small bars for clarity
        if abs(data[-1]) < 0.25:
            continue

        # Adjust y-position for label
        y_position = (bottom - 0.5 * data)[-1]
        if i == 0:
            y_position = (0.5 * data)[-1]

    ###################

    ax.set_ylabel("Average system cost [billion USD/year]")
    ax.set_ylim([0, 16])
    ax.set_xlim([0, 3])  # <-- Adjusted xlim
    ax.set_xticks([0.5, 1.5, 2.5])  # Set positions of the tick marks
    ax.set_xticklabels(["AC\nline", "wo\nep", "w\nep"], fontsize=10)
    ax.grid(True, axis="y", color="k", linestyle="dotted")


##########
=======
        text = ax.text(
            2, y_position, opts["nice_names"].get(ind, ind)
        )
        texts.append(text)

    ax.set_ylabel("Average system cost [billion US$/year]")
    ax.set_ylim([0, 15]) #opts.get("costs_max", 80)])
    ax.set_xlim([0, 2])
    ax.set_xticklabels([])
    ax.grid(True, axis="y", color="k", linestyle="dotted")


>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
<<<<<<< HEAD

        snakemake = mock_snakemake(
            'plot_network_sa',
=======
        snakemake = mock_snakemake(
            'plot_network',
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
            **{
                'model_file': 'val-2Gt-IRP',
                'regions': '27-supply',
                'resarea': 'redz',
                'll': 'copt',
<<<<<<< HEAD
                'opts': 'Co2L-Ep-1H',  ####Co2L-1H_inv_2,Co2L-Ep-1H,Co2L-1H_base
=======
                'opts': 'Co2L-1H_base',
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
                'attr': 'p_nom',
                'ext': 'png'
            }
        )
    configure_logging(snakemake)

    model_setup = (pd.read_excel(snakemake.input.model_file,
                                 sheet_name='model_setup',
                                 index_col=[0])
    .loc[snakemake.wildcards.model_file])

    set_plot_style()

    config, wildcards = snakemake.config, snakemake.wildcards

    map_figsize = config["plotting"]["map"]["figsize"]
    map_boundaries = config["plotting"]["map"]["boundaries"]

    n = load_network_for_plots(
        snakemake.input.network, snakemake.input.model_file, config, model_setup.costs
    )

    scenario_opts = wildcards.opts.split("-")

    fig, ax = plt.subplots(
        figsize=map_figsize, subplot_kw={"projection": ccrs.PlateCarree()}
    )
    plot_map(n, config["plotting"], ax=ax, attribute=wildcards.attr)

    fig.savefig(snakemake.output.only_map, dpi=150, bbox_inches="tight")

<<<<<<< HEAD
    ax1 = fig.add_axes([-0.115, 0.5, 0.2, 0.2])
    plot_total_energy_pie(n, config["plotting"], ax=ax1)

    ax2 = fig.add_axes([-0.075, 0.1, 0.15, 0.4])
=======
    ax1 = fig.add_axes([-0.115, 0.625, 0.2, 0.2])
    plot_total_energy_pie(n, config["plotting"], ax=ax1)

    ax2 = fig.add_axes([-0.075, 0.1, 0.1, 0.45])
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
    plot_total_cost_bar(n, config["plotting"], ax=ax2)

    ll = wildcards.ll
    ll_type = ll[0]
    ll_factor = ll[1:]
    lbl = dict(c='line cost', v='line volume')[ll_type]
    amnt = '{ll} x today\'s'.format(ll=ll_factor) if ll_factor != 'opt' else 'optimal'
<<<<<<< HEAD
    # fig.suptitle('Expansion to {amount} {label} at {regions} regions'
    #            .format(amount=amnt, label=lbl, regions=wildcards.regions))
    fig.savefig(snakemake.output.ext, transparent=True, bbox_inches='tight')
=======
    fig.suptitle('Expansion to {amount} {label} at {regions} regions'
                .format(amount=amnt, label=lbl, regions=wildcards.regions))
    fig.savefig(snakemake.output.ext, transparent=True, bbox_inches='tight')
>>>>>>> 837b413abbe71e2ec8092ee04261ac6966eeb173
