#! /usr/bin/gnuplot

### Retrieve from other data files
n_grid    = system("awk '/n_grid/  {print $3}' start.py | head -1")
n_basis   = system("awk '/n_basis/ {print $3}' start.py | head -1")
grid_file = "grid_".n_basis.".dat"
dat_file  = "dataset_".n_basis.".dat"

# Define styles
PTS = 'pointtype 7 pointsize 1.5 linecolor rgb "blue"'

### Plot png

# Initial stuff
set terminal png size 1600,800
set output "rbf.png"
set multiplot layout 1,2

# Prepare table with map
set table 'map.dat'
splot grid_file u 1:2:3
unset table

# Prepare isolines with map
set dgrid3d n_grid,n_grid
set contour base
set cntrparam levels discrete 0
unset surface
set table 'iso.dat'
splot grid_file u 1:2:3
unset table

# Plot approx heatmap with samples
reset
unset key
unset xtics
unset ytics
set xrange [-1:1]
set yrange [-1:1]
set title "rbf level-set, basis size=".n_basis
unset colorbox
set cbrange [-1:1]
set palette rgb 21,22,23
plot 'map.dat' with image, \
     'iso.dat' w l lw 3,   \
     dat_file  u 1:2 w p @PTS

# Plot surface
reset
unset key
set title "rbf level-set, basis size=".n_basis
set dgrid3d n_grid,n_grid
set hidden3d
set surface
set contour surface
set cntrparam levels discrete 0
set isosamples 50
splot grid_file u 1:2:3 w l notitle, \
      grid_file u 1:2:3 w l nosurf lw 3