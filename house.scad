CubePoints = [
[0.0, 0.0, 0.0],
[20.0, 0.0, 0.0],
[20.0, 4.0, 0.0],
[15.0, 4.0, 0.0],
[15.0, 10.0, 0.0],
[10.0, 10.0, 0.0],
[10.0, 15.0, 0.0],
[20.0, 15.0, 0.0],
[20.0, 35.0, 0.0],
[0.0, 35.0, 0.0],
[0.0, 0.0, 5.0],
[20.0, 0.0, 5.0],
[20.0, 4.0, 5.0],
[15.0, 4.0, 5.0],
[15.0, 10.0, 5.0],
[10.0, 10.0, 5.0],
[10.0, 15.0, 5.0],
[20.0, 15.0, 5.0],
[20.0, 35.0, 5.0],
[0.0, 35.0, 5.0],
[18.0, 2.0, 7.82842712474619],
[10.0, 25.0, 19.14213562373095],
[13.0, 2.0, 7.82842712474619],
[5.0, 20.0, 12.071067811865476],
[10.0, 5.0, 12.071067811865476],
[5.0, 5.0, 12.071067811865476]];

CubeFaces = [
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
[0, 10, 11, 1],
[1, 11, 12, 2],
[2, 12, 13, 3],
[3, 13, 14, 4],
[4, 14, 15, 5],
[5, 15, 16, 6],
[6, 16, 17, 7],
[7, 17, 18, 8],
[8, 18, 19, 9],
[9, 19, 10, 0],
[10, 25, 24, 22, 20, 11],
[11, 20, 12],
[12, 20, 22, 13],
[13, 22, 24, 14],
[14, 24, 25, 15],
[15, 25, 23, 16],
[16, 23, 21, 17],
[17, 21, 18],
[18, 21, 19],
[19, 21, 23, 25, 10]];

polyhedron( CubePoints, CubeFaces );