def plot_panel(rows, plot, x_values, description, show_legend=False):
    '''
    Plot one panel with log-log axes.
    
    Arguments:
        rows (list): List of tuples (label, y_values, color, marker)
        plot (matplotlib.axes.Axes): The axes object of the panel
        x_values (numpy.ndarray): The count grid for the x-axis
        description (str): The annotation text of the panel
    '''
    for label, y_values, color, marker in rows:
        plot.loglog(x_values, y_values, linestyle='-', linewidth=2.0, markersize=9, markeredgewidth=1.0, markeredgecolor='black', color=color, marker=marker, label=label)
    
    plot.set_xscale('log')
    plot.set_yscale('log')
    
    plot.set_ylabel(r'$\mathrm{Cumulative\ time\ (s)}$', fontsize=25)
    plot.set_xlabel(r'$\mathrm{Number\ of\ evaluations}$', fontsize=25)
    
    plot.grid(True, which='both', alpha=0.50)
    if show_legend:
        plot.legend(loc='lower right', fontsize=25)
    plot.text(0.02, 0.98, description, fontsize=25, verticalalignment='top', transform=plot.transAxes)