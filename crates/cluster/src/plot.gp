# 1. Terminal Setup
set title "3D PCA: SrTiO3 Domain Analysis" 

# 2. Data Handling
set datafile separator "," 
# If your CSV has no headers, we'll manually label the axes

# 3. Visual Styling
set grid 
set xyplane at 0          # Adjusts the "floor" of the 3D box 
set view 60, 30, 1.2, 1   # Initial camera angle (elevation, azimuth, scale, z-scale) 

# 4. Color Mapping
# We will color the points by their Z-value (PC3) to show depth
set palette rgbformulae 33,13,10  # A nice 'rainbow' 
# or 'ocean' style palette [cite: 2]

# 5. The 3D Plot Command (splot)
# 'using 1:2:3:3' -> X=col1, Y=col2, Z=col3, Color=col3 
# Increased ps from 0.8 to 1.5 to make the dots larger
splot "pca_3d.csv" using 1:2:3:3 with points pt 7 ps 1.5 palette notitle 

# 6. Keep the window open
pause mouse close
