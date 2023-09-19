CubePoints = [
  [  0,  0,  0 ],//0
  [ 12,  0,  0 ],
  [ 12,  5,  0 ],
  [  6,  5,  0 ],
  [  6, 10,  0 ],
  [ 12, 10,  0 ],
  [ 12, 20,  0 ],
  [ 6, 20,  0 ],
  [  6, 15,  0 ],
  [  0, 15,  0 ]];

CubeFaces = [
  [0,1,2,3,4,5,6,7,8,9,10]];


roof_angle = 45;

roof_height = 10;

polyhedron( CubePoints, CubeFaces );