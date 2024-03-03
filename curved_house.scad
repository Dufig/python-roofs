CubePoints = [
[0.0, 2.91, 0.0],
[4.0, 0.0, 0.0],
[8.36, 6.0, 0.0],
[10.4, 6.0, 0.0],
[14.0, 2.4, 0.0],
[14.0, 0.0, 0.0],
[12.0, -2.0, 0.0],
[15.0, -5.0, 0.0],
[20.0, 0.0, 0.0],
[20.0, 8.0, 0.0],
[18.0, 10.0, 0.0],
[5.16, 10.0, 0.0],
[0.0, 2.91, 5.0],
[4.0, 0.0, 5.0],
[8.36, 6.0, 5.0],
[10.4, 6.0, 5.0],
[14.0, 2.4, 5.0],
[14.0, 0.0, 5.0],
[12.0, -2.0, 5.0],
[15.0, -5.0, 5.0],
[20.0, 0.0, 5.0],
[20.0, 8.0, 5.0],
[18.0, 10.0, 5.0],
[5.16, 10.0, 5.0],
[3.4541092259287667, 3.4545289552869995, 8.497394321510283],
[15.0, -2.0, 8.0],
[16.585786437626904, 6.585786437626908, 8.695518130045144],
[6.418462978159132, 7.527642928797327, 7.774216746204775],
[16.121320343559642, -0.8786796564403572, 7.296100594190538],
[7.341811768050446, 8.0, 7.244260964255394],
[17.0, 1.2426406871192857, 8.247176600877182],
[11.228427124746192, 8.0, 7.164784400584788],
[17.0, 3.6426406871192842, 8.247176600877182],
[16.016652224137044, 6.016652224137047, 9.1409008082146]];

CubeFaces = [
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
[0, 12, 13, 1],
[1, 13, 14, 2],
[2, 14, 15, 3],
[3, 15, 16, 4],
[4, 16, 17, 5],
[5, 17, 18, 6],
[6, 18, 19, 7],
[7, 19, 20, 8],
[8, 20, 21, 9],
[9, 21, 22, 10],
[10, 22, 23, 11],
[11, 23, 12, 0],
[12, 24, 13],
[13, 24, 27, 29, 14],
[14, 29, 31, 15],
[15, 31, 33, 32, 16],
[16, 32, 30, 28, 17],
[17, 28, 25, 18],
[18, 25, 19],
[19, 25, 28, 30, 20],
[20, 30, 32, 33, 26, 21],
[21, 26, 22],
[22, 26, 33, 31, 29, 27, 23],
[23, 27, 24, 12]];

polyhedron( CubePoints, CubeFaces );