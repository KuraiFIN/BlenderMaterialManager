import bpy
import math
from mathutils import Vector

import bmesh

objects = bpy.context.selected_objects[:]
#Change above with: global objects; objects = val;

"""
COMMON
"""

def copy_object(object, do_link=True):
    obj = object.copy()
    if do_link: obj.data = object.data
    else: obj.data = object.data.copy()
    collection = bpy.context.scene.collection
    collection.objects.link(obj)
    #for modifier in obj.modifiers: if modifier.type == 'ARRAY': obj.modifiers.remove(modifier)
    return obj

def place_object(new_object, location, parent=None):
    new_object.location = location
    if parent is not None:
        new_object.name = parent.name + '_' + new_object.data.name
        new_object.parent = parent
        #new_object.matrix_parent_inverse = parent.matrix_world.inverted()
        #Above is commented, which gives it parent's offset from parent, so we need:
        new_object.location = Vector((0,0,0))
    return new_object

class Vert():
    def __init__(self, co, index):
        self.co = co
        self.index = index
        self.src = None
    def __repr__(self):
        return 'Vertex<' + str(self.co) + ', index=' + str(self.index) + '>'
    def bv(bvert):
        out = Vert(bvert.co, bvert.index)
        out.src = bvert
        return out
    def vec(var):
        return getattr(var, 'co', var)
    def mix(lhs, rhs, start_bias = True):
        co = (lhs.co + rhs.co) / 2.0
        index = lhs.index if start_bias else rhs.index
        out = Vert(co, index)
        out.src = lhs.src if start_bias else rhs.src
        return out
    def mix_index(ilhs, irhs, start_bias = True):
        output = (ilhs, irhs, start_bias)
        return output
    def mix_by(vectors, instruction):
        lhs, rhs, start_bias = instruction
        return Vert.mix(vectors[lhs], vectors[rhs], start_bias)
    def pick(lhs, rhs, ref, furthest = True):
        seq = (lhs.co - ref.co).length > (rhs.co - ref.co).length
        most = lhs if seq else rhs
        least = rhs if seq else lhs
        return most if furthest else least
    def pick_index(vectors, ilhs, irhs, ref, furthest = True):
        lhs, rhs = vectors[ilhs], vectors[irhs]
        seq = (lhs.co - ref.co).length > (rhs.co - ref.co).length
        most = ilhs if seq else irhs
        least = irhs if seq else ilhs
        main = most if furthest else least
        output = (main, ilhs, irhs)
        return output
    def pick_by(vectors, ref, instruction):
        main, lhs, rhs = instruction
        return vectors[main]
    def average(verts):
        mean = Vector((0.0, 0.0, 0.0))
        for vert in verts:
            #Or hasattr(vert, 'co') returns bool
            #Or try: something; except AttributeError: otherthing;
            co = getattr(vert, 'co', vert)
            mean += co
        mean /= len(verts)
        return Vert(mean, len(verts))
    def flatten_box(vectors, axis, pick_over_mix = True, start_or_furthest = True):
        if len(vectors) < 8: return vectors
        v = vectors
        pm = pick_over_mix
        sf = start_or_furthest
        avg = Vert.average(vectors)
        p = Vert.pick
        m = Vert.mix
        if axis == 0:
            #merge 0+1, 2+3, 4+5, 6+7
            return [p(v[0],v[1],avg,sf), p(v[2],v[3],avg,sf), p(v[4],v[5],avg,sf), p(v[6],v[7],avg,sf)] if pm else [m(v[0],v[1],sf), m(v[2],v[3],sf), m(v[4],v[5],sf), m(v[6],v[7],sf)]
        elif axis == 1:
            #merge 0+2, 1+3, 4+6, 5+7
            return [p(v[0],v[2],avg,sf), p(v[1],v[3],avg,sf), p(v[4],v[6],avg,sf), p(v[5],v[7],avg,sf)] if pm else [m(v[0],v[2],sf), m(v[1],v[3],sf), m(v[4],v[6],sf), m(v[5],v[7],sf)]
        elif axis == 2:
            #merge 0+4, 1+5, 2+6, 3+7
            return [p(v[0],v[4],avg,sf), p(v[1],v[5],avg,sf), p(v[2],v[6],avg,sf), p(v[3],v[7],avg,sf)] if pm else [m(v[0],v[4],sf), m(v[1],v[5],sf), m(v[2],v[6],sf), m(v[3],v[7],sf)]
        return vectors
    def flatten_multiple(vectors_list, axis, pick_over_mix = True, start_or_furthest = True):
        if len(vectors_list) == 0: return vectors_list
        vectors = vectors_list[0]#main
        for vectors_alt in vectors_list:
            if len(vectors_alt) != len(vectors): return vectors_list
        v = vectors
        pm = pick_over_mix
        sf = start_or_furthest
        avg = Vert.average(vectors)
        p = Vert.pick
        pi = Vert.pick_index
        pb = Vert.pick_by
        m = Vert.mix
        mi = Vert.mix_index
        mb = Vert.mix_by
        if axis == 0:
            instruct = [pi(v,0,1,avg,sf), pi(v,2,3,avg,sf), pi(v,4,5,avg,sf),pi(v,6,7,avg,sf)] if pm else [mi(0,1,sf), mi(2,3,sf), mi(4,5,sf), mi(6,7,sf)]
            return [ [pb(vl,avg,i) if pm else mb(vl,i) for i in instruct] for vl in vectors_list]
        elif axis == 1:
            instruct = [pi(v,0,2,avg,sf), pi(v,1,3,avg,sf), pi(v,4,6,avg,sf),pi(v,5,7,avg,sf)] if pm else [mi(0,2,sf), mi(1,3,sf), mi(4,6,sf), mi(5,7,sf)]
            return [ [pb(vl,avg,i) if pm else mb(vl,i) for i in instruct] for vl in vectors_list]
        elif axis == 2:
            instruct = [pi(v,0,4,avg,sf), pi(v,1,5,avg,sf), pi(v,2,6,avg,sf),pi(v,3,7,avg,sf)] if pm else [mi(0,4,sf), mi(1,5,sf), mi(2,6,sf), mi(3,7,sf)]
            return [ [pb(vl,avg,i) if pm else mb(vl,i) for i in instruct] for vl in vectors_list]
        return vectors_list
    def distance(origin, vert, use_x = True, use_y = True, use_z = True):
        o = Vert.vec(origin)
        v = Vert.vec(vert)
        ro = Vector((o.x if use_x else 0.0, o.y if use_y else 0.0, o.z if use_z else 0.0))
        rv = Vector((v.x if use_x else 0.0, v.y if use_y else 0.0, v.z if use_z else 0.0))
        return (ro - rv).length
    def distance_on(origin, vert, axis = 0):
        b = [False, False, False]
        if -1 < axis < 3: b[axis] = True
        return Vert.distance(origin, vert, b[0], b[1], b[2])
    def distance_not_on(origin, vert, axis):
        b = [True, True, True]
        if -1 < axis < 3: b[axis] = False
        return Vert.distance(origin, vert, b[0], b[1], b[2])
    def nearest(origin, verts, use_x = True, use_y = True, use_z = True):
        dist = 0.0
        out = -1
        for i in range(len(verts)):
            d = Vert.distance(origin, verts[i], use_x, use_y, use_z)
            if out == -1 or d < dist:
                out = i
                dist = d
        return out
    def nearest_on(origin, verts, axis = 0):
        b = [False, False, False]
        if -1 < axis < 3: b[axis] = True
        return Vert.nearest(origin, verts, b[0], b[1], b[2])
    def nearest_not_on(origin, verts, axis):
        b = [True, True, True]
        if -1 < axis < 3: b[axis] = False
        return Vert.nearest(origin, verts, b[0], b[1], b[2])
    def select_set(self, v):
        if self.src is not None: self.src.select_set(v)
        return None

def angle2D(vec1, vec2):
    dot = vec1.x * vec2.x + vec1.y * vec2.y
    det = vec1.x * vec2.y - vec1.y * vec2.x
    return math.atan2(det, dot)

class Ref:
    def __init__(self, ref, member, value):
        self.ref = ref
        self.member = member
        self.value = value
    def __repr__(self):
        return 'Ref<(' + str(self.ref) + ').' + str(self.member) + ' = ' + str(self.value) + '>'



"""
ORGANIZATION
"""

def get_prop(object, key, default = ''):
    prop = object.get(key)
    if prop is None: return default
    return prop

def object_name(object, mode = 'SELF'):
    m = mode.upper()
    if m == 'SELF' or len(m) == 0: return object.name
    elif object.type == m: return object.data.name
    elif m == 'PROP': return get_prop(object, 'name', object.name)
    return mode

def fullname(object, name, prefix = '', sep = ':'):
    output = name
    if object is not None and len(prefix) != 0: output = object_name(object, prefix) + sep + name
    return output

def set_name(object, mode, value):
    v = value
    vm = value.upper()
    if vm == 'SELF': v = object.name
    elif vm == object.type: v = object.data.name
    elif vm == 'PROP': v = get_prop(object, 'name', object.name)
    m = mode.upper()
    if m == 'SELF': object.name = v
    elif m == object.type:
        object.data.name = object.data.name[0:-1] * 2# stupidly necessary
        object.data.name = v
    elif m == 'PROP': object['name'] = v
    return object
def set_names(mode, value):
    for object in objects:
        set_name(object, mode, value)
    return True



"""
UV MAP LAYERS
"""

def find_map(object, name, prefix_mode = ''):
    if object.type != 'MESH': return -1
    n = name
    if len(prefix_mode) != 0: n = object_name(object, prefix_mode) + ':' + name
    for i, map in enumerate(object.data.uv_layers):
        if map.name == n: return i
    return -1

def has_map(object, name, prefix_mode = ''):
    return find_map(object, name, prefix_mode) != -1

def select_map(key, prefix_mode = '', render = True):
    output = True
    for object in objects:
        if object.type != 'MESH': continue
        n = str(key)
        if len(prefix_mode) != 0: n = object_name(object, prefix_mode) + ':' + n
        found = False
        for i, map in enumerate(object.data.uv_layers):
            #print(object.name, map.name)
            #map = layer, map.data = x[], x = struct{bool select, float[2] uv}
            valid_int = type(key) == int and i == key
            if valid_int or n == map.name:
                object.data.uv_layers.active_index = i
                if render: map.active_render = True
                found = True
                break
        if not found: output = False
    return output

def append_map(name, prefix_mode = '', select = False):
    for object in objects:
        n = name
        if len(prefix_mode) != 0: n = object_name(object, prefix_mode) + ':' + name
        if object.type != 'MESH' or has_map(object, n): continue
        new_map = object.data.uv_layers.new(name=n)
    if select: select_map(name, prefix_mode)
    #complement: remove()
    return None



"""
VERTEX GROUP LAYERS
"""
def find_group(object, name, prefix_mode = ''):
    if object.type != 'MESH': return -1
    n = name
    if len(prefix_mode) != 0: n = object_name(object, prefix_mode) + ':' + name
    for i, vg in enumerate(object.vertex_groups):
        if vg.name == n: return i
    return -1

def has_group(object, name, prefix_mode = ''):
    return find_group(object, name, prefix_mode) != -1

def select_group(key, prefix_mode = '', render = True):
    output = True
    for object in objects:
        if object.type != 'MESH': continue
        n = str(key)
        if len(prefix_mode) != 0: n = object_name(object, prefix_mode) + ':' + n
        found = False
        for i, vg in enumerate(object.vertex_groups):
            valid_int = type(key) == int and i == key
            if valid_int or n == vg.name:
                object.vertex_groups.active_index = i
                #if render
                found = True
                break
        if not found: output = False
    return output

def append_group(name, prefix_mode = '', select = False):
    for object in objects:
        n = name
        if len(prefix_mode) != 0: n = object_name(object, prefix_mode) + ':' + name
        if object.type != 'MESH' or has_group(object, n): continue
        new_group = object.vertex_groups.new(name=n)
    if select: select_group(name, prefix_mode)
    #complement: remove()
    return None

def in_group(object, name, prefix_mode = ''):
    output = []
    vgi = find_group(object, name, prefix_mode)
    if vgi == -1 or object.type != 'MESH': return output
    #Assuming Blender stores indices consistently
    output = [v.index for v in object.data.vertices if vgi in [vg.group for vg in v.groups] ]
    return output


"""
OBJECT SPACE BOUNDARY
"""

def object_vbox(object, transform = False):
    matrix = object.matrix_world
    corners = [Vector(corner) for corner in object.bound_box]
    centre = Vector((0.0, 0.0, 0.0))
    if transform:
        centre = matrix @ centre
        for i, corner in enumerate(corners):
            corners[i] = matrix @ corner
    left = right = forward = backward = downward = upward = centre
    for corner in corners:
        if corner.x < left.x: left = Vector((corner.x, centre.y, centre.z))
        if corner.x > right.x: right = Vector((corner.x, centre.y, centre.z))
        if corner.y < forward.y: forward = Vector((centre.x, corner.y, centre.z))
        if corner.y > backward.y: backward = Vector((centre.x, corner.y, centre.z))
        if corner.z < downward.z: downward = Vector((centre.x, centre.y, corner.z))
        if corner.z > upward.z: upward = Vector((centre.x, centre.y, corner.z))
    vecs = [left, right, forward, backward, downward, upward]
    return vecs

#Gets both mirror and array!
def object_bbox(object, transform = False):
    matrix = object.matrix_world
    corners = [Vector(corner) for corner in object.bound_box]
    if transform:
        for i, corner in enumerate(corners):
            corners[i] = matrix @ corner
    left = right = corners[0].x
    forward = backward = corners[0].y
    downward = upward = corners[0].z
    for corner in corners:
        if corner.x < left: left = corner.x
        if corner.x > right: right = corner.x
        if corner.y < forward: forward = corner.y
        if corner.y > backward: backward = corner.y
        if corner.z < downward: downward = corner.z
        if corner.z > upward: upward = corner.z
    floats = [left, right, forward, backward, downward, upward]
    return floats

def object_dimensions(object, transform = False):
    f = object_bbox(object, transform)
    return Vector(( abs(f[1] - f[0])/1, abs(f[3] - f[2])/1, abs(f[5] - f[4])/1 ))

def is_mirrored(object):
    for modifier in object.modifiers:
        if modifier.type == 'MIRROR': return True
    return False
def mirror_axis(object):
    for modifier in object.modifiers:
        if modifier.type == 'MIRROR':
            axis = modifier.use_axis
            for i in range(len(axis)):
                if axis[i]: return i
        #modifier.use_mirror_u = True
        #modifier.mirror_offset_u (flipping point offset from 0.5)
        #modifier.offset_u (general offset)
        #use_mirror_udim I think is for texture space repeat/clamp or in this case flip
    return -1
def set_mirror_offset(object, value, u = True):
    for modifier in object.modifiers:
        if modifier.type == 'MIRROR':
            if u: modifier.mirror_offset_u = value#flipping point offset from 0.5
            else: modifier.mirror_offset_v = value
            #if u: modifier.offset_u = value
            #else: modifier.offset_v = value
    return None

def array_amount(object):
    for modifier in object.modifiers:
        if modifier.type == 'ARRAY':
            if modifier.relative_offset_displace[0] > 1.1: return 1
            return modifier.count
    return 1
def array_width_single(object):
    for modifier in object.modifiers:
        if modifier.type == 'ARRAY':
            if modifier.relative_offset_displace[0] > 1.1:
                spc = (modifier.relative_offset_displace[0] - 1.0) * (modifier.count - 1)
                geo = modifier.count * 1.0
                all = spc + geo
                return ((geo / all) * object.dimensions.x) / modifier.count
            return object.dimensions.x / modifier.count
    return object.dimensions.x

def bm_uvbox(bm):
    bbox = {}
    uv_layers = bm.loops.layers.uv.verify()
    boundsMin = Vector((99999999.0,99999999.0))
    boundsMax = Vector((-99999999.0,-99999999.0))
    boundsCenter = Vector((0.0,0.0))
    #Was for face in island
    for face in bm.faces:
        for loop in face.loops:
            uv = loop[uv_layers].uv
            boundsMin.x = min(boundsMin.x, uv.x)
            boundsMin.y = min(boundsMin.y, uv.y)
            boundsMax.x = max(boundsMax.x, uv.x)
            boundsMax.y = max(boundsMax.y, uv.y)
    bbox['min'] = Vector((boundsMin))
    bbox['max'] = Vector((boundsMax))
    boundsCenter.x = (boundsMax.x + boundsMin.x)/2
    boundsCenter.y = (boundsMax.y + boundsMin.y)/2
    bbox['center'] = boundsCenter
    return bbox




"""
OBJECT RELATIONS
"""

def get_neighbor(object, bbox_index):
    i = 0
    even = bbox_index % 2 == 0
    if even: i = bbox_index + 1
    else: i = bbox_index - 1
    vbox = object_vbox(object, True)
    compare = vbox[bbox_index]
    error = 0.5
    for o in bpy.context.view_layer.objects:
        if object == o: continue
        o_vbox = object_vbox(o, True)
        d = abs((o_vbox[i] - compare).length)
        if d < error:
            return o
    #TODO: return hanging trim etc
    return None

def get_wall(sample, select = False):
    
    #object.rotation_euler.to_quaternion().to_axis_angle() => Return type: (Vector, float) pair
    
    return None




def to_bmesh(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    return bm
def to_mesh(bm, mesh):
    bm.to_mesh(mesh)
    bm.free()
    return mesh

#bv is expected from bm.verts
def uv_first(uv_layer, bv):
    for loop in bv.link_loops:
        uv_data = loop[uv_layer]
        return uv_data.uv
    return None
def uv_average(uv_layer, bv):
    sum = Vector((0.0, 0.0))
    count = 0.0
    for loop in bv.link_loops:
        sum += loop[uv_layer].uv
        count += 1.0
    if count > 0.0: return sum / count
    return sum
def uv_all(uv_layer, bv):
    output = []
    for loop in bv.link_loops:
        output.append(loop[uv_layer].uv)
    return output
def offset_uv_vert(uv_layer, bv, x = 0.0, y = 0.0, first_only = False):
    for loop in bv.link_loops:
        loop[uv_layer].uv += Vector((x, y))
        if first_only: return bv
    return bv
def offset_uv_island(uv_layer, bface, x = 0.0, y = 0.0):
    for vertex in bface.loops:
        vertex[uv_layer].uv += Vector((x, y))
    return bface

def uv_squareness(uv_layer, bface):
    #use bv.link_faces[read_only] to get connected faces
    uvs = []
    for vertex in bface.loops:
        ##uvs.append(uv_average(uv_layer, vertex.vert))
        uv = vertex[uv_layer].uv # only worry about the uv in THIS face
        uvs.append(Vector((uv.x, uv.y, 0.0)))
    out = 0.0
    for i in range(len(uvs)):
        e = (uvs[i] - uvs[i - 1]).normalized()
        a = angle2D(uvs[i], uvs[i - 1]) * 57.29578
        if a > 180.0: a = abs(a - 360.0)
        if a > 90.0: a -= 90.0
        r = 45.0 - a if a < 45.0 else a - 90.0 # higher the closer to +/-45
        out += abs(r / 45.0)
    return out / len(uvs)
def uv_squareness_avg(uv_layer, bv):
    sum = 0.0
    count = 0.0
    for face in bv.link_faces:
        sum += uv_squareness(uv_layer, face)
        count += 1.0
    return sum / count

def nearest_uv_bias(uv_layer, origin, verts, use_x = True, use_y = True, use_z = True):
    dist = 0.0
    out = -1
    for i in range(len(verts)):
        #large = extra square-like
        sqr = min(uv_squareness_avg(uv_layer, verts[i].src) - 1.0, 1.0)
        #if sqr <= 0.00001: continue
        d = Vert.distance(origin, verts[i], use_x, use_y, use_z)
        d += 1.0 - sqr # now large = extra round
        if out == -1 or d < dist:
            out = i
            dist = d
    #return out if out != -1 else Vert.nearest(origin, verts, use_x, use_y, use_z)
    return out
def nearest_uv_bias_on(uv_layer, origin, verts, axis = 0):
    b = [False, False, False]
    if -1 < axis < 3: b[axis] = True
    return nearest_uv_bias(uv_layer, origin, verts, b[0], b[1], b[2])
    
def vector_axis_mean(vectors, axis=0):
    sum = 0.0
    if len(vectors) == 0: return mean
    for v in vectors:
        sum += v.co[axis]
    return sum / len(vectors)
def vector_axis_polarities(vectors, axis=0):
    if len(vectors) == 0: return []
    mean = vector_axis_mean(vectors, axis)
    return [True if v.co[axis] >= mean else False for v in vectors]
def vectors_sorted(vectors):
    output = vectors[:]
    x = vector_axis_polarities(vectors, 0)
    y = vector_axis_polarities(vectors, 1)
    z = vector_axis_polarities(vectors, 2)
    
    mods = [False for v in vectors]
    for i in range(len(vectors)):
        addZ = 4 if z[i] else 0
        addX = 1 if x[i] else 0
        addY = 2 if y[i] else 0
        index = addZ + addX + addY
        if mods[index]:
            if addY == 0: index += 2
            else: index -= 2
        output[index] = vectors[i]
        mods[index] = True
    return output

def get_bounding_verts(object, bm):
    #First, test if we have a vgroup
    vgi = in_group(object, 'bbox')
    if len(vgi) != 0:
        out = vectors_sorted([Vert.bv(bm.verts[v]) for v in vgi])
        print('Bounding BVerts:', out)
        return out
    b = object_bbox(object, False)
    vbox = [Vector((b[0],b[2],b[4])), Vector((b[1],b[2],b[4])), Vector((b[0],b[3],b[4])), Vector((b[1],b[3],b[4])),
            Vector((b[0],b[2],b[5])), Vector((b[1],b[2],b[5])), Vector((b[0],b[3],b[5])), Vector((b[1],b[3],b[5]))]
    floats = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    output = [None,None,None,None,None,None,None,None]
    uvs = [None, None, None, None,None,None,None,None]
    uv_layer = bm.loops.layers.uv.active#bm.loops.layers.uv.verify()
    ##Double Hash is for converting UV coords into new bm as vertices
    ##vert_index = uvbm.verts/faces.layers.int.new("index")#idk what this does
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    for face in bm.faces:
        #Necessary for vert.select to have effect:
        face.select = False#bpy.ops.mesh.select_all(action = 'DESELECT')
        #offset_uv_island(uv_layer, face, 0.01, 0.01)#test works
        ##fverts = []
        for vertex in face.loops:
            #Attributes pulled from vertex:
            uv = vertex[uv_layer].uv
            uv = Vector((uv.x, uv.y, 0.0))
            #print("Loop UV: %f, %f" % uv[:])
            vert = vertex.vert
            vert.select = False
            ##v = bm.verts.new((uv.x, uv.y, 0))
            ##v[vert_index] = loop.vert.index
            ##fverts.append(v)
            #print("Loop Vert: (%f,%f,%f)" % vert.co[:])
            for c, corner in enumerate(vbox):
                d = (vert.co - corner).length
                if output[c] is None or d < floats[c]:
                    floats[c] = d
                    output[c] = Vert.bv(vert)
                    uvs[c] = Vert(uv, vert.index)
        ##f = bmesh.ops.contextual_create(bm, geom=fverts)["faces"].pop()
        ##f[face_index] = face.index
    
    #output_flat = Vert.flatten_box(output, 1)
    #for out in output_flat:
    #    out.select_set(True)
    uvs, output = Vert.flatten_multiple([uvs, output], 1)
    for i, out in enumerate(output):
        out.select_set(True)
    
    #Mesh Mirror
    ma = mirror_axis(object)
    if ma != -1:
        nearest = nearest_uv_bias_on(uv_layer, Vector((0.0,0.0,0.0)), output, ma)
        bv = bm.verts[output[nearest].index]
        uv_avg = uv_average(uv_layer, bv)
        set_mirror_offset(object, (uv_avg.x - 0.5) * 2.0, True)
        print('NearestUV:', uv_avg)
    
    print('Distances:', floats)
    print('Bounding BVerts:', output)
    #print('Boudning BVFlat:', output_flat)
    print('Polarities X:', vector_axis_polarities(output))
    print('UV:', uvs)




"""
MATERIALS
"""
def find_material(object, name, prefix_mode = ''):
    #object.active_material/_index
    #object.material_slots[0]{link = 'OBJECT'||'DATA', material, name}
    if object.type != 'MESH': return -1
    n = name
    if len(prefix_mode) != 0: n = object_name(object, prefix_mode) + ':' + name
    for i, mat in enumerate(object.material_slots):
        if mat.name == n: return i
    return -1
def has_material(object, name, prefix_mode = ''):
    return find_material(object, name, prefix_mode) != -1
def find_material_any(object, name, prefix_mode = ''):
    output = []
    if object.type != 'MESH': return output
    n = name
    if len(prefix_mode) != 0: n = object_name(object, prefix_mode) + ':' + name
    for i, mat in enumerate(object.material_slots):
        if mat.name.startswith(n): output.append(i)
    return output
def has_material_any(object, name, prefix_mode = ''):
    return len(find_material_any(object, name, prefix_mode)) != 0
def select_material(key, prefix_mode = '', override_from = None):
    output = True
    for object in objects:
        if object.type != 'MESH': continue
        n = str(key)
        if len(prefix_mode) != 0: n = object_name(object, prefix_mode) + ':' + n
        found = False
        for i, mat in enumerate(object.material_slots):
            valid_int = type(key) == int and i == key
            if valid_int or n == mat.name:
                object.active_material_index = i
                #if override_from
                found = True
                break
        if not found: output = False
    return output

def append_material(name, prefix_mode = '', select = False):
    for object in objects:
        n = name
        if len(prefix_mode) != 0: n = object_name(object, prefix_mode) + ':' + name
        if object.type != 'MESH' or has_material(object, n): continue
        new_mat = bpy.data.materials.new(n)
        new_mat.use_nodes = True
        nodes = new_mat.node_tree.nodes
        new_tex = nodes.new('ShaderNodeTexImage')
        socket = nodes['Principled BSDF'].inputs[0]
        new_mat.node_tree.links.new(socket, new_tex.outputs[0])
        object.data.materials.append(new_mat)
        #object.material_slots[object.active_material_index].material = new_mat
    if select: select_material(name, prefix_mode)
    #complement: remove()
    return None


#Expected use: object.material_slots[i].material if material and material.node_tree
def get_textures(material):
    output = []
    if not material or not material.node_tree: return output
    for node in material.node_tree.nodes:
        if node.type == 'TEX_IMAGE' and node.image: output.append(node.image.name)
    return output

def in_material(object, name, prefix_mode = ''):
    output = []
    mat = find_material(object, name, prefix_mode)
    if mat == -1 or object.type != 'MESH': return output
    #Assuming Blender stores indices consistently
    output = [f.index for f in object.data.polygons if f.material_index == mat]
    return output
def reassign_material(object, bm, dst_key, src_key):
    ds, ss = type(dst_key) != int, type(src_key) != int or src_key < 0
    dst = find_material(object, str(dst_key)) if ds else dst_key
    src = find_material_any(object, str(dst_key)) if ss else [src_key]
    if dst == -1 or len(src) == -1: return False
    if bm is None:
        for face in object.data.polygons:
            if face.material_index in src: face.material_index = dst
        object.data.update()
    else:
        for face in bm.faces:
            if face.material_index in src: face.material_index = dst
    return True




def test1(object):
    bm = to_bmesh(object.data)
    #Also Selects!!!
    get_bounding_verts(object, bm)
    reassign_material(object, bm, 0, 1)
    to_mesh(bm, object.data)


#append_map('some_uv', 'PROP')
#select_map(0, 'PROP')
#set_names('MESH', 'SELF')

#print(object_bbox(objects[0], True))
#print(get_neighbor(objects[0], 0))
test1(objects[0])

append_group('vgroupB')
print(in_group(objects[0], 'vgroupB'))
append_material('new_material')


"""
TODO STILL:
    *Align UV islands spanning 0-1 for objects in wall
    *Add function to bake from one uvmap to the next (I had one lying around here somewhere...)
    
"""
