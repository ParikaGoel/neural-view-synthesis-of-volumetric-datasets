import plyfile
import numpy as np


def write_ply(file, vertices, faces, edges):
    """
        Writes the given vertices and faces to PLY.
        :param vertices: vertices as tuples of (x, y, z, r, g, b) coordinates
        :type vertices: [(float)] -> list of tuples
        :param faces: faces as tuples of (num_vertices, vertex_id_1, vertex_id_2, ...)
        :type faces: [(int)]
        """

    num_vertices = len(vertices)
    num_faces = len(faces)
    num_edges = len(edges)

    assert num_vertices > 0

    if num_faces == 0:
        print("No face information provided. Saving point cloud. To visualize voxelized mesh, please provide face information.")

    with open(file, 'w') as fp:
        fp.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            element face %d
            property list uchar int vertex_index
            element edge %d
            property int vertex1
            property int vertex2
            property uchar red
            property uchar green
            property uchar blue
            end_header
            ''' % (num_vertices, num_faces, num_edges))

        for vert in vertices:
            if len(vert) == 3:
                color = (0, 169, 255)
                fp.write('%f %f %f %d %d %d\n' % vert+color)
            elif len(vert) == 6:
                fp.write('%f %f %f %d %d %d\n' % tuple(vert))
            else:
                print('Error: Incorrect number of properties in a vertex. Expected 3 or 6 entries\n')
                return

        for face in faces:
            fp.write('3 %d %d %d\n' % tuple(face))
        
        for edge in edges:
            fp.write('%d %d %d %d %d\n' % tuple(edge))


def read_ply(filename):
    """
    Reads vertices and faces from a ply file.
    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """
    plydata = plyfile.PlyData().read(filename)

    vertices = [list(x) for x in plydata['vertex']]  # Element 0 is the vertex
    faces = [x.tolist() for x in plydata['face'].data['vertex_index']]  # Element 1 is the face
    return vertices, faces


def grid_to_mesh(grid_lst, ply_file, min_bound=np.array([-0.5, -0.5, -0.5]), voxel_scale=1/32):
    """
    Utility function to visualize the voxel grid
    :param grid_lst: grid coordinates with their corresponding color information
    :param ply_file: filename to export mesh information to
    :param min_bound: world coordinates corresponding to grid coordinate (0,0,0)
    :param voxel_scale: length of one voxel in world system
    If not provided, we consider a 32^3 unit length voxel grid
    min_bound = np.array([-grid_size / 2, -grid_size / 2, -grid_size / 2])
    voxel_scale = grid_size / voxel_resolution
    """
    cube_verts = np.array([[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
                           [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]])  # 8 points

    cube_faces = np.array([[0, 1, 2], [2, 3, 0], [1, 5, 6], [6, 2, 1], [7, 6, 5], [5, 4, 7],
                           [4, 0, 3], [3, 7, 4], [4, 5, 1], [1, 0, 4], [3, 2, 6], [6, 7, 3]])  # 6 faces (12 triangles)

    verts = []
    faces = []
    curr_vertex = 0

    for data in grid_lst:
        grid_color = np.array((data[3], data[4], data[5])).astype(int)

        for cube_vert in cube_verts:
            vertex = (cube_vert * 0.45 + np.array([data[0], data[1], data[2]])).astype(float)
            vertex *= voxel_scale
            vertex += min_bound
            vertex = np.append(vertex, grid_color)
            verts.append(list(vertex))

        for cube_face in cube_faces:
            face = curr_vertex + cube_face
            faces.append(list(face))

        curr_vertex += len(cube_verts)

    write_ply(ply_file, verts, faces)