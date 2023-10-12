CubePoints = [
  [  0,  0,  0 ],//0
  [ 15,  0,  0 ],
  [ 20, 10,  0 ],
  [ 25, 25,  0 ],
  [ -3, 23,  0 ],
  [ 10, 10,  0 ]];

CubeFaces = [
  [0,1,2,3,4,5]];


roof_angle = 45;

roof_height = 5;

polyhedron( CubePoints, CubeFaces );