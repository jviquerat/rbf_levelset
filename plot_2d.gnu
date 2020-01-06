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
reset
set print "-"
set terminal png size 2400,800
unset key
set output "rbf.png"
set multiplot layout 1,3

# Plot approx heatmap with samples
set title "sampling, basis size=".n_basis
set cbrange [-1:1]
unset xtics
unset ytics
plot grid_file u 1:2:3 with image, \
     dat_file  u 1:2 w p @PTS
set xtics
set ytics

# Plot contour
set view map
unset surface
set dgrid3d n_grid,n_grid
set contour surface
set cntrparam levels discrete 0
set title "rbf level-set, basis size=".n_basis
splot grid_file u 1:2:3 w l lw 3, \
      dat_file  u 1:2:(1) w p @PTS

# Plot surface
unset view
set hidden3d
set surface
set isosamples 50
set title "RBF, basis size=".n_basis
splot grid_file u 1:2:3 w l notitle, grid_file u 1:2:3 w l nosurf lw 3