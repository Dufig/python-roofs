CubePoints = [
  [  0,  0,  0 ],
  [ 12,  0,  0 ],  
  [ 12,  8,  0 ],
  [  5,  8,  0 ],
  [  5, 15,  0 ],
  [  0, 15,  0 ],
  [  0,  0,  5 ],
  [ 12,  0,  5 ],  
  [ 12,  8,  5 ],
  [  5,  8,  5 ],
  [  5, 15,  5 ],
  [  0, 15,  5 ]];
  
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
