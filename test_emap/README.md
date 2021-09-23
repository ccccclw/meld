### Grid force feature 
---
<!-- - [x] linear interpolation (reference)
- [x] linear interpolation test(reference) -->
$V_{xyz} = m\frac{1}{L^3}(V_{000}(x_{max}-x)(y_{max}-y)(z_{max}-z) + \\
\hspace{31pt} V_{100}x(y_{max}-y)(z_{max}-z) + \\
\hspace{31pt} V_{010}(x_{max}-x)y(z_{max}-z) + \\
\hspace{31pt} V_{001}(x_{max}-x)(y_{max}-y)z + \\
\hspace{31pt} V_{101}x(y_{max}-y)z + \\
\hspace{31pt} V_{011}(x_{max}-x)yz + \\
\hspace{31pt} V_{110}xy(z_{max}-z) + \\
\hspace{31pt} V_{111}xyz)$  
where $m$ is the atom mass and $L$ is the grid length, $x_{max}, y_{max}, z_{max}$ are grid maximum value (the right point) and $(x,y,z)$ is atom coordinate.



