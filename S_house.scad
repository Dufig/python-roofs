CubePoints = [
  [ -2,  0,  0 ],//0
  [ 12,  0,  0 ],
  [ 12,  6,  0 ],
  [  8,  6,  0 ],
  [  8, 15,  0 ],
  [ -6, 15,  0 ],
  [ -6, 10,  0 ],
  [ -2, 10,  0 ]];


roof_height = -5;

roof_angle = 45;

CubeFaces = [
  [0,1,2,3,4,5,6,7]];
  
polyhedron( CubePoints, CubeFaces );
