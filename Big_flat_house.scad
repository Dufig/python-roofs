CubePoints = [
  [  0,  0,  0 ],//0
  [ 20,  0,  0 ],
  [ 20,  4,  0 ],
  [ 15,  4,  0 ],
  [ 15, 10,  0 ],
  [ 10, 10,  0 ],//
  [ 10, 15,  0 ],
  [ 20, 15,  0 ],
  [ 20, 35,  0 ],
  [  0, 35,  0 ]];

CubeFaces = [
  [0,1,2,3,4,5,6,7,8,9]];


roof_angle = 45;

roof_height = 5;

polyhedron( CubePoints, CubeFaces );