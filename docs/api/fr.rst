====================
Free recall analysis
====================

.. currentmodule:: psifr.fr

Managing data
~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    table_from_lists
    check_data
    merge_free_recall
    merge_lists
    filter_data
    reset_list
    split_lists
    pool_index
    block_index

Recall probability
~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    spc
    pnr

Intrusions
~~~~~~~~~~

.. autosummary::
    :toctree: api/

    pli_list_lag

Transition probability
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    lag_crp
    lag_crp_compound
    category_crp
    distance_crp

Transition rank
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    lag_rank
    distance_rank
    distance_rank_shifted

Clustering
~~~~~~~~~~

.. autosummary::
    :toctree: api/

    category_clustering

Plotting
~~~~~~~~

.. autosummary::
    :toctree: api/

    plot_raster
    plot_spc
    plot_lag_crp
    plot_distance_crp
    plot_swarm_error
