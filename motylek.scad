CubePoints = [
  [  1,  1,  0 ],
  [  8, -3,  0 ],  
  [ 12,  8,  0 ],
  [  5,  8,  0 ],
  [ -2, 15,  0 ],
  [ -9,  4,  0 ],
  [  1,  1,  5 ],
  [  8, -3,  5 ],  
  [ 12,  8,  5 ],
  [  5,  8,  5 ],
  [ -2, 15,  5 ],
  [ -9,  4,  5 ]];
  
CubeFaces = [
  [0,1,2,3,4,5],
  [4,5,11,10],
  [3,4,10,9],
  [2,3,9,8],
  [1,2,8,7],
  [0,1,7,6],
  [0,5,11,6],
  [6,7,8,9,10,11]];
  
polyhedron( CubePoints, CubeFaces );