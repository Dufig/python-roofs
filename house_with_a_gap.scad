CubePoints = [
  [  0, -1,  0 ],//0
  [ 18, -1,  0 ],  
  [ 18,  6,  0 ],
  [ 12,  6,  0 ],
  [ 12,  2,  0 ],
  [  6,  2,  0 ],//5
  [  6, 12,  0 ],
  [ 12, 12,  0 ],
  [ 12,  8,  0 ],
  [ 18,  8,  0 ],
  [ 18, 15,  0 ],//10
  [  0, 15,  0 ]];
  
CubeFaces = [
  [0,1,2,3,4,5,6,7,8,9,10,11]];

roof_angle = 45;

roof_height = 10;
  
polyhedron( CubePoints, CubeFaces );
