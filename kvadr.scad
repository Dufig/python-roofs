CubePoints = [
  [  0,  0,  0 ],
  [ 10,  0,  0 ],
  [ 10,  7,  0 ],
  [  0,  7,  0 ],
  [  0,  0,  5 ],
  [ 10,  0,  5 ],
  [ 10,  7,  5 ],
  [  0,  7,  5 ]];
  
CubeFaces = [
  [0,1,2,3],
  [4,5,1,0],
  [7,6,5,4],
  [5,6,2,1],
  [6,7,3,2],
  [7,4,0,3]];
  
polyhedron( CubePoints, CubeFaces );
