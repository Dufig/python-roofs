CubePoints = [
  [  0,  0,  0 ],//0
  [ 12,  0,  0 ],  
  [ 12,  5,  0 ],
  [  6,  5,  0 ],
  [  6, 10,  0 ],
  [ 12, 10,  0 ],
  [ 12, 15,  0 ],
  [  0, 15,  0 ]];
  
CubeFaces = [
  [0,1,2,3,4,5,6,7]];

roof_angle = 45;

roof_height = 5;
  
polyhedron( CubePoints, CubeFaces );
