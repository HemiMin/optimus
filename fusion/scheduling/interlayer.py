import math
import numpy as np
from scipy.optimize import minimize
import copy

from collections import namedtuple

from .. import util
from .layer import ConvLayer, LocalRegionLayer
from .network import Network
from .resource import Resource
from . import schedule_generator
from . import loop_enum as le


class InterLayerReuse(object):
    """
    Inter-layer reuse.
    """

    SchedIndex = namedtuple('SchedIndex', ['sp_idx', 'tm_idx'])
    Scale = namedtuple('Scale', ['s_h', 's_w'])
    MinSize = namedtuple('MinSize', ['h', 'w'])
    print('SchedIndex:',SchedIndex)
    print('Scale:',Scale)
    print('MinSize:',MinSize)

    def __init__(self, network, fusion_group, resource, loop_lower_bound, topological_order=True,
                 z_fusion=False, d_fusion=False, womincost=False):
        if not isinstance(network, Network):
            raise TypeError('InterLayerReuse: network must be '
                            'a Network instance.')

        if not isinstance(fusion_group, list):
            raise TypeError('InterLayerReuse: fusion_group must be '
                            'a list.')

        if not isinstance(resource, Resource):
            raise TypeError('InterLayerPipeline: resource must be '
                            'a Resource instance.')

        self.network = network
        self.fusion_group = fusion_group
        self.resource = resource
        self.loop_lower_bound = loop_lower_bound
        self.topological_order = topological_order
        self.z_fusion = z_fusion
        self.d_fusion = d_fusion
        self.womincost = womincost

        self.valid = self._prepare()
        if not self.valid:
            return

        # self._calc_sched_dag()
        #
        # self.valid = self._init_alternate_pair()
        # if not self.valid:
        #     return

    def sched(self, mode):
        if self.valid:
            self._calc_sched_dag()
            self.valid = self._init_alternate_pair(mode)

    def _prepare(self):
        self.firsts = []
        self.lasts = []
        self.ext_inputs_dict = dict()
        self.ext_outputs = set()
        self.fused_weight_size = 0
        self.fused_input_size = 0
        self.fused_output_size = 0

        print('\n||prepare')
        if self.d_fusion:
            for layer in self.fusion_group:
                if len(self.network.nexts(layer)) > 1 or len(self.network.prevs(layer)) > 1:
                    self.valid = False
                    return False

        for layer in self.fusion_group:
            print(layer)
            tmp = tuple()
            for nx in self.network.nexts(layer):
                if nx not in self.fusion_group:
                    tmp += (nx, )
                    self.ext_outputs.add(layer)
            if tmp == self.network.nexts(layer):
                self.lasts.append(layer)

            tmp = tuple()
            for pre in self.network.prevs(layer):
                if pre not in self.fusion_group:
                    tmp += (pre, )
                    if pre not in self.ext_inputs_dict:
                        self.ext_inputs_dict[pre] = [layer]
                    else:
                        self.ext_inputs_dict[pre].append(layer)
            if tmp == self.network.prevs(layer):
                if isinstance(self.network[layer], LocalRegionLayer):
                    return False
                self.firsts.append(layer)
            if isinstance(self.network[layer], ConvLayer):
                self.fused_weight_size += self.network[layer].total_filter_size
                print('fused_weight_size:', self.fused_weight_size)

        for ip in self.ext_inputs_dict:
            print('input:',ip)
            if ip is None:
                self.fused_input_size += self.network[self.network.INPUT_LAYER_KEY].total_ofmap_size
            else:
                self.fused_input_size += self.network[ip].total_ofmap_size
            print('fused_input_size:', self.fused_input_size)
        for op in self.ext_outputs:
            self.fused_output_size += self.network[op].total_ofmap_size
            print('output:',op)
            print('fused_output_size:', self.fused_output_size)
        return True

    def _calc_sched_dag(self):

        # The DAG vertex list in the topological order.
        if self.topological_order:
            self.dag_vertex_list = self._topological_order()
        else:
            self.dag_vertex_list = self.fusion_group

        # Make a directory from layer name to DAG vertex index.
        self.dag_vertex_dict = {}

        for vidx, layer_name in enumerate(self.dag_vertex_list):
            assert layer_name not in self.dag_vertex_dict
            self.dag_vertex_dict[layer_name] = vidx

        # The previous and next relationship of the DAG vertices.
        self.dag_prev_dict = dict((vidx, set()) for vidx
                                  in range(len(self.dag_vertex_list)))
        self.dag_next_dict = dict((vidx, set()) for vidx
                                  in range(len(self.dag_vertex_list)))

        for layer_name in self.fusion_group:
            vidx = self.dag_vertex_dict[layer_name]

            # Previous layers.
            for p in self.network.prevs(layer_name):
                if not p or p not in self.fusion_group:
                    continue
                pvidx = self.dag_vertex_dict[p]
                if pvidx != vidx:
                    self.dag_prev_dict[vidx].add(pvidx)

            # Next layers.
            for n in self.network.nexts(layer_name):
                if not n or n not in self.fusion_group:
                    continue
                nvidx = self.dag_vertex_dict[n]
                if nvidx != vidx:
                    self.dag_next_dict[vidx].add(nvidx)

        self.ext_inputs_idx = {}
        for vidx, layer_name in enumerate(self.ext_inputs_dict.keys()):
            assert layer_name not in self.dag_vertex_dict
            self.ext_inputs_idx[layer_name] = vidx + len(self.dag_vertex_list)

    def _topological_order(self):

        # The visited layers in the DFS order.
        visited = []
        # The unseen pending layers.
        unseen = set(self.fusion_group)
        # The layers that have been seen, but not visited due to unvisited
        # previous layers.
        seen = set()

        def _dfs(vertex):
            assert vertex not in seen
            if vertex in visited:
                return

            unseen.discard(vertex)
            seen.add(vertex)

            next_vertices = []

            for n in reversed(self.network.nexts(vertex)):
                if n and n not in next_vertices and n in unseen:
                    next_vertices.append(n)

            for nv in next_vertices:
                _dfs(nv)

            visited.append(vertex)
            seen.remove(vertex)

        # Start from the first layers.
        for v in self.firsts:
            _dfs(v)
        assert not unseen
        assert not seen

        return list(reversed(visited))

    def ordered_layer_list(self):

        return list(sum(self.dag_vertex_list, tuple()))

    def _init_scale(self):
        print('\n||init_scale')
        scale_tmp = [None for _ in self.dag_vertex_list]

        for idx, l in enumerate(self.dag_vertex_list):
            layer = self.network[l]
            print(str(idx),':',l,':',layer )
            if l in self.firsts:
                scale_tmp[idx] = [layer.hstd, layer.wstd]
                print('scale_tmp1:',scale_tmp[idx])
                continue

            max_hs, max_ws = 0, 0
            for src in self.dag_prev_dict[idx]:
                src_scale = scale_tmp[src]
                assert src_scale
                max_hs = src_scale[0] if src_scale[0] > max_hs else max_hs
                max_ws = src_scale[1] if src_scale[1] > max_ws else max_ws
            scale_tmp[idx] \
                = [max_hs * layer.hstd, max_ws * layer.wstd]
            print('scale_tmp2:',scale_tmp[idx])

        self.scale = [None for _ in self.dag_vertex_list]

        last_h = []
        last_w = []
        for l in self.lasts:
            idx = self.dag_vertex_dict[l]
            last_h.append(scale_tmp[idx][0])
            last_w.append(scale_tmp[idx][1])
        s_h = util.lcm(*last_h)
        s_w = util.lcm(*last_w)

        for l in reversed(self.dag_vertex_list):
            idx = self.dag_vertex_dict[l]
            layer = self.network[l]
            print('scale l:',str(idx),':',l,':',layer )
            if l in self.lasts:
                self.scale[idx] = \
                    InterLayerReuse.Scale(s_h / scale_tmp[idx][0], s_w / scale_tmp[idx][1])
                continue

            print('scale_tmp3:',scale_tmp[idx])
            s_h_tmp, s_w_tmp = None, None
            for dst_idx in self.dag_next_dict[idx]:
                dst = self.dag_vertex_list[dst_idx]
                dst_layer = self.network[dst]
                dst_scale = self.scale[dst_idx]
                print('\tdst_layer(',dst,'):',dst_layer)
                assert dst_scale
                if s_h_tmp is None and s_w_tmp is None:
                    s_h_tmp, s_w_tmp = dst_layer.hstd * dst_scale.s_h, dst_layer.wstd * dst_scale.s_w
                else:
                    assert s_h_tmp == dst_layer.hstd * dst_scale.s_h \
                           and s_w_tmp == dst_layer.wstd * dst_scale.s_w
            self.scale[idx] = \
                InterLayerReuse.Scale(s_h_tmp, s_w_tmp)
            print('scale_tmp4:',scale_tmp[idx])

        self.minSize = [None for _ in self.dag_vertex_list]
        for l in reversed(self.dag_vertex_list):
            idx = self.dag_vertex_dict[l]
            layer = self.network[l]
            print('minsize l:',str(idx),':',l,':',layer )
            if l in self.lasts:
                self.minSize[idx] = InterLayerReuse.MinSize(self.scale[idx].s_h, self.scale[idx].s_w)
                print('min_size1:',self.minSize[idx])
                continue

            h_tmp, w_tmp = None, None
            for dst_idx in self.dag_next_dict[idx]:
                dst = self.dag_vertex_list[dst_idx]
                dst_layer = self.network[dst]
                dst_minsize = self.minSize[dst_idx]
                print('\tdst_layer(',dst,'):',dst_layer)
                print('\tdst_minsize:',dst_minsize)
                assert dst_minsize
                if isinstance(dst_layer, LocalRegionLayer):
                    hreg, wreg = dst_layer.hreg, dst_layer.wreg
                else:
                    hreg, wreg = dst_layer.hfil, dst_layer.wfil
                print('\tdst_hreg,wreg:',hreg,',',wreg)
                if h_tmp is None and w_tmp is None:
                    h_tmp = (dst_minsize.h - 1) * dst_layer.hstd + hreg
                    w_tmp = (dst_minsize.w - 1) * dst_layer.wstd + wreg
                    h_tmp = layer.hofm if h_tmp > layer.hofm else h_tmp
                    w_tmp = layer.wofm if w_tmp > layer.wofm else w_tmp
                else:
                    if (dst_minsize.h - 1) * dst_layer.hstd + hreg > h_tmp:
                        h_tmp = (dst_minsize.h - 1) * dst_layer.hstd + hreg
                        h_tmp = layer.hofm if h_tmp > layer.hofm else h_tmp
                    if (dst_minsize.w - 1) * dst_layer.wstd + wreg > w_tmp:
                        w_tmp = (dst_minsize.w - 1) * dst_layer.wstd + wreg
                        w_tmp = layer.wofm if w_tmp > layer.wofm else w_tmp
            self.minSize[idx] = InterLayerReuse.MinSize(h_tmp, w_tmp)
            print('minsize2:',self.minSize[idx])

    def _init_alternate_pair_optimus(self):
        self._init_scale()
        irrelevant = [le.D, le.R, le.K, le.C]
        print('irrelevant:',irrelevant)
        self.loop_block = [None for _ in self.dag_vertex_list]
        self.loop_order = [None for _ in self.dag_vertex_list]

        self.min_feature_footprint, self.is_full_buffer, self.add_one_line_footprint, self.min_required_mem_size = self._alternate()

        print('\n||init_alternate_pair_optimus')
        if self.is_full_buffer is None:
            print('is_full_buffer None')
            return False

        level = self.resource.buffer_levels() - 2
        s = self.resource.buffer(level).capacity * self. resource.paras[level].count
        print('s(',level,'):',s)

        if s < self.min_feature_footprint + self.fused_weight_size:
            return False

        if s >= self.fused_weight_size + self.min_feature_footprint:
            self.sfil_fit = True
            self.tile_num = 1

            h_m = self.network.input_layer().hofm
            w_m = self.network.input_layer().wofm
            for l in self.lasts:
                print('  last l:', l)
                if self.network[l].hofm < h_m:
                    h_m = self.network[l].hofm
                if self.network[l].wofm < w_m:
                    w_m = self.network[l].wofm
            print('h_m:',h_m,'w_m:',w_m)

            s = s - self.fused_weight_size
            line_num = max(math.floor((s - self.min_feature_footprint) /
                                  self.add_one_line_footprint),0) + 1
            print('line_num:', line_num)
            if line_num > h_m:
                h = h_m
                b = int(max(s // ((h_m - 1) * self.add_one_line_footprint + self.min_feature_footprint),
                            self.network.input_layer().nimg))
            else:
                h = line_num
                b = 1
            print('h:',h,'b:',b)

            H = [None for _ in self.dag_vertex_list]
            print('calc H')
            for l in reversed(self.dag_vertex_list):
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                print(idx,':',l,':',layer)
                if l in self.lasts:
                    H[idx] = int(self.scale[idx].s_h * h)
                    print('H[',idx,']:',H[idx])
                    continue

                h_tmp = None
                for dst_idx in self.dag_next_dict[idx]:
                    dst = self.dag_vertex_list[dst_idx]
                    dst_layer = self.network[dst]
                    dst_h = H[dst_idx]
                    print('  dstl:',dst,':',dst_layer)
                    print('  dst_h:',dst_h)
                    assert dst_h is not None
                    if isinstance(dst_layer, LocalRegionLayer):
                        hreg = dst_layer.hreg
                    else:
                        hreg = dst_layer.hfil
                    if h_tmp is None:
                        h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                    else:
                        if (dst_h - 1) * dst_layer.hstd + hreg > h_tmp:
                            h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                H[idx] = math.floor(h_tmp)
                print('H[',idx,']:',H[idx])

            print('loopblock & looporder')
            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                print(idx,':',l,':',layer)
                if isinstance(layer, LocalRegionLayer):
                    continue
                loop_lower_bound = self.loop_lower_bound(layer)
                kk = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nofm)
                print('kk:',kk)
                k = layer.nofm if self.is_full_buffer[idx] else kk
                c = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nifm)
                print('k:',k,'(is_full_buffer:',self.is_full_buffer[idx],')')
                print('c:',c)

                for pre in self.dag_prev_dict[idx]:
                    if self.is_full_buffer[pre]:
                        c = layer.nifm
                        print(' pre:', self.dag_vertex_list[pre])
                        print(' c:',c,'(is_full_buffer:',self.is_full_buffer[pre],')')

                if k < layer.nofm and c < layer.nifm:
                    c = layer.nifm
                    print('c:',c,'(k(',k,')<layer.nofm(',layer.nofm,') and c(',c,')<layer.nifm(',layer.nifm,')')
                self.loop_block[idx] = [layer.wfil, layer.hfil, c, layer.wofm, H[idx], k, b]
                print('loop_block:[kw:',layer.wfil,',kh:',layer.hfil,',ic:',c,',ow:',layer.wofm,',oh:',H[idx],',oc:',k,',b:',b,']')
                self.loop_order[idx] = schedule_generator.loop_order_generator(layer, self.loop_block[idx], irrelevant)
                print('loop_order:',self.loop_order[idx])

        else:
            if self.z_fusion or self.d_fusion:
                return False
            self.sfil_fit = False
            h_m = self.network.input_layer().hofm
            w_m = self.network.input_layer().wofm
            for l in self.lasts:
                print('  last l:', l)
                if self.network[l].hofm < h_m:
                    h_m = self.network[l].hofm
                if self.network[l].wofm < w_m:
                    w_m = self.network[l].wofm
            print('h_m:',h_m,'w_m:',w_m)

            line_num = math.floor((s - self.min_feature_footprint) / self.add_one_line_footprint) + 1
            print('line_num:', line_num)
            if line_num > h_m:
                h = h_m
                b = int(min(s // ((h_m - 1) * self.add_one_line_footprint + self.min_feature_footprint),
                            self.network.input_layer().nimg))
            else:
                h = line_num
                b = 1
            print('h:',h,'b:',b)

            H = [None for _ in self.dag_vertex_list]
            print('calc H')
            for l in reversed(self.dag_vertex_list):
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                print(idx,':',l,':',layer)
                if l in self.lasts:
                    H[idx] = int(self.scale[idx].s_h * h)
                    print('H[',idx,']:',H[idx])
                    continue

                h_tmp = None
                for dst_idx in self.dag_next_dict[idx]:
                    dst = self.dag_vertex_list[dst_idx]
                    dst_layer = self.network[dst]
                    dst_h = H[dst_idx]
                    print('  dstl:',dst,':',dst_layer)
                    print('  dst_h:',dst_h)
                    assert dst_h is not None
                    if isinstance(dst_layer, LocalRegionLayer):
                        hreg = dst_layer.hreg
                    else:
                        hreg = dst_layer.hfil
                    if h_tmp is None:
                        h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                    else:
                        if (dst_h - 1) * dst_layer.hstd + hreg > h_tmp:
                            h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                H[idx] = math.floor(h_tmp)
                print('H[',idx,']:',H[idx])

            print('loopblock & looporder')
            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                print(idx,':',l,':',layer)
                if isinstance(layer, LocalRegionLayer):
                    print('continue')
                    continue
                loop_lower_bound = self.loop_lower_bound(layer)

                k = layer.nofm if self.is_full_buffer[idx] \
                    else min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nofm)
                c = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nifm)
                print('k:',k,'(is_full_buffer:',self.is_full_buffer[idx],')')
                print('c:',c)

                for pre in self.dag_prev_dict[idx]:
                    if self.is_full_buffer[pre]:
                        c = layer.nifm
                        print(' pre:', pre)
                        print('c:',c,'(is_full_buffer:',self.is_full_buffer[pre],')')
                if k < layer.nofm and c < layer.nifm:
                    c = layer.nifm
                    print('c:',c,'(k(',k,')<layer.nofm(',layer.nofm,') and c(',c,')<layer.nifm(',layer.nifm,')')
                self.loop_block[idx] = [layer.wfil, layer.hfil, c, layer.wofm, H[idx], k, b]
                print('loop_block:[kw:',layer.wfil,',kh:',layer.hfil,',ic:',c,',ow:',layer.wofm,',oh:',H[idx],',oc:',k,',b:',b,']')
                self.loop_order[idx] = \
                    schedule_generator.loop_order_generator(layer, self.loop_block[idx], irrelevant)
                print('loop_order:',self.loop_order[idx])

                self.tile_num = math.ceil(h_m * self.network.input_layer().nimg / (b * h))
                print('tile_num:',self.tile_num)
                print('ceil(h_m(',h_m,')*nimg(',self.network.input_layer().nimg,')/(b(',b,')*h(',h,')))')
        self.q = self.fused_weight_size * self.tile_num + self.fused_input_size + self.fused_output_size

        if self.network.net_name != "SqueezeNet":
            p2 = self.resource.access_cost[2]
            p1 = self.resource.access_cost[1]

            q0, q1, q2 = p1[0], p1[1], p2[2] + p1[2]

            f_args = (q0, q1, q2, b, h_m)
            fun = self.fun(f_args)
            c_args = (b, h_m, self.idx_dict)
            con = self.con(c_args)
            x0 = [1 for _ in range(len(set(self.idx_dict.values())) + 1)]
            if b > 1:
                x0[0] = b
            else:
                x0[0] = h
            for idx in self.idx_dict:
                if idx < len(self.dag_vertex_list):
                    layer = self.network[self.dag_vertex_list[idx]]
                    if isinstance(layer, LocalRegionLayer):
                        continue
                    loop_lower_bound = self.loop_lower_bound(layer)
                    x0[self.idx_dict[idx]] = loop_lower_bound.k

            x0 = np.asarray(x0)
            print(f'x0:{x0}')
            res = minimize(fun, x0, method='COBYLA', constraints=con)
            print(f'resx0:{res.x}')

            if res.success:
                if b > 1:
                    b = math.ceil(res.x[0])
                    h = h_m
                else:
                    b = 1
                    h = math.ceil(res.x[0])
                print(f'minimizeh:{h}')
                H = [None for _ in self.dag_vertex_list]
                for l in reversed(self.dag_vertex_list):
                    idx = self.dag_vertex_dict[l]
                    layer = self.network[l]
                    print(str(idx),':',l,':',layer )
                    if l in self.lasts:
                        H[idx] = int(self.scale[idx].s_h * h)
                        print(f'H[{idx}]:{H[idx]}')
                        continue

                    h_tmp = None
                    for dst_idx in self.dag_next_dict[idx]:
                        print(f'h_tmp:{h_tmp}')
                        dst = self.dag_vertex_list[dst_idx]
                        dst_layer = self.network[dst]
                        dst_h = H[dst_idx]
                        print(f'dst_layer:{dst_layer}')
                        print(f'dst_h:{dst_h}')
                        assert dst_h is not None
                        if isinstance(dst_layer, LocalRegionLayer):
                            hreg = dst_layer.hreg
                        else:
                            hreg = dst_layer.hfil
                        if h_tmp is None:
                            h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                            print(f'h_tmp({h_tmp})=min((dst_h({dst_h})-1)*dst_layer_hstd({dst_layer.hstd})+hreg({hreg}),hofm({layer.hofm})')
                        else:
                            if (dst_h - 1) * dst_layer.hstd + hreg > h_tmp:
                                h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                                print(f'h_tmp({h_tmp})=min((dst_h({dst_h})-1)*dst_layer_hstd({dst_layer.hstd})+hreg({hreg}),hofm({layer.hofm})')
                    H[idx] = math.floor(h_tmp)
                    print(f'H[{idx}]:{H[idx]}')

                for l in self.dag_vertex_list:
                    idx = self.dag_vertex_dict[l]
                    layer = self.network[l]
                    if isinstance(layer, LocalRegionLayer):
                        continue

                    if idx in self.idx_dict:
                        k = res.x[self.idx_dict[idx]]
                    else:
                        k = layer.nofm

                    if self.dag_prev_dict[idx] and list(self.dag_prev_dict[idx])[0] in self.idx_dict:
                        c = self.idx_dict[list(self.dag_prev_dict[idx])[0]]
                    else:
                        c = layer.nifm
                    self.loop_block[idx] = \
                        [layer.wfil, layer.hfil, math.ceil(c), layer.wofm, H[idx], math.ceil(k), b]

                    self.loop_order[idx] = \
                        schedule_generator.loop_order_generator(layer, self.loop_block[idx], irrelevant)

            q = self.fused_weight_size * self.network.input_layer().nimg * h_m * q2 / (b * h) \
                + self.fused_input_size * p2[0] + self.fused_output_size * p2[1]
            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if isinstance(layer, LocalRegionLayer):
                    continue
                q += (q1 * layer.nifm * layer.total_ofmap_size / self.loop_block[idx][2])
                q += (q0 * layer.nofm * layer.total_ifmap_size / self.loop_block[idx][5])
            self.q = q
            print('q:', q)
            self.tile_num = math.ceil(h_m * self.network.input_layer().nimg / (b * h))

        return True

    def _init_alternate_pair_others(self, mode):
        self._init_scale()
        irrelevant = [le.D, le.R, le.K, le.C]
        self.loop_block = [None for _ in self.dag_vertex_list]
        self.loop_order = [None for _ in self.dag_vertex_list]

        self.min_feature_footprint, self.is_full_buffer, self.add_one_line_footprint = self._alternate()

        if self.is_full_buffer is None:
            return False

        level = self.resource.buffer_levels() - 2
        s = self.resource.buffer(level).capacity * self. resource.paras[level].count

        if s < self.min_feature_footprint + self.fused_weight_size:
            return False

        if s >= self.fused_weight_size + self.min_feature_footprint:
            self.sfil_fit = True
            self.tile_num = 1

            h_m = self.network.input_layer().hofm
            w_m = self.network.input_layer().wofm
            for l in self.lasts:
                if self.network[l].hofm < h_m:
                    h_m = self.network[l].hofm
                if self.network[l].wofm < w_m:
                    w_m = self.network[l].wofm

            s = s - self.fused_weight_size
            line_num = math.floor((s - self.min_feature_footprint) /
                                  self.add_one_line_footprint) + 1
            if line_num > h_m:
                h = h_m
                b = int(max(s // ((h_m - 1) * self.add_one_line_footprint + self.min_feature_footprint),
                            self.network.input_layer().nimg))
            else:
                h = line_num
                b = 1

            H = [None for _ in self.dag_vertex_list]
            for l in reversed(self.dag_vertex_list):
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if l in self.lasts:
                    H[idx] = int(self.scale[idx].s_h * h)
                    continue

                h_tmp = None
                for dst_idx in self.dag_next_dict[idx]:
                    dst = self.dag_vertex_list[dst_idx]
                    dst_layer = self.network[dst]
                    dst_h = H[dst_idx]
                    assert dst_h is not None
                    if isinstance(dst_layer, LocalRegionLayer):
                        hreg = dst_layer.hreg
                    else:
                        hreg = dst_layer.hfil
                    if h_tmp is None:
                        h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                    else:
                        if (dst_h - 1) * dst_layer.hstd + hreg > h_tmp:
                            h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                H[idx] = math.floor(h_tmp)

            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if isinstance(layer, LocalRegionLayer):
                    continue
                loop_lower_bound = self.loop_lower_bound(layer)
                kk = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nofm)
                k = layer.nofm if self.is_full_buffer[idx] else kk
                c = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nifm)

                for pre in self.dag_prev_dict[idx]:
                    if self.is_full_buffer[pre]:
                        c = layer.nifm

                if k < layer.nofm and c < layer.nifm:
                    c = layer.nifm
                self.loop_block[idx] = [layer.wfil, layer.hfil, c, layer.wofm, H[idx], k, b]
                self.loop_order[idx] = schedule_generator.loop_order_generator(layer, self.loop_block[idx], irrelevant)

        else:
            if self.z_fusion or self.d_fusion:
                return False
            self.sfil_fit = False
            h_m = self.network.input_layer().hofm
            w_m = self.network.input_layer().wofm
            for l in self.lasts:
                if self.network[l].hofm < h_m:
                    h_m = self.network[l].hofm
                if self.network[l].wofm < w_m:
                    w_m = self.network[l].wofm

            line_num = math.floor((s - self.min_feature_footprint) / self.add_one_line_footprint) + 1
            if line_num > h_m:
                h = h_m
                b = int(min(s // ((h_m - 1) * self.add_one_line_footprint + self.min_feature_footprint),
                            self.network.input_layer().nimg))
            else:
                h = line_num
                b = 1

            H = [None for _ in self.dag_vertex_list]
            for l in reversed(self.dag_vertex_list):
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if l in self.lasts:
                    H[idx] = int(self.scale[idx].s_h * h)
                    continue

                h_tmp = None
                for dst_idx in self.dag_next_dict[idx]:
                    dst = self.dag_vertex_list[dst_idx]
                    dst_layer = self.network[dst]
                    dst_h = H[dst_idx]
                    assert dst_h is not None
                    if isinstance(dst_layer, LocalRegionLayer):
                        hreg = dst_layer.hreg
                    else:
                        hreg = dst_layer.hfil
                    if h_tmp is None:
                        h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                    else:
                        if (dst_h - 1) * dst_layer.hstd + hreg > h_tmp:
                            h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                H[idx] = math.floor(h_tmp)

            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if isinstance(layer, LocalRegionLayer):
                    continue
                loop_lower_bound = self.loop_lower_bound(layer)

                k = layer.nofm if self.is_full_buffer[idx] \
                    else min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nofm)
                c = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nifm)

                for pre in self.dag_prev_dict[idx]:
                    if self.is_full_buffer[pre]:
                        c = layer.nifm
                if k < layer.nofm and c < layer.nifm:
                    c = layer.nifm
                self.loop_block[idx] = [layer.wfil, layer.hfil, c, layer.wofm, H[idx], k, b]
                self.loop_order[idx] = \
                    schedule_generator.loop_order_generator(layer, self.loop_block[idx], irrelevant)

                self.tile_num = math.ceil(h_m * self.network.input_layer().nimg / (b * h))
        self.q = self.fused_weight_size * self.tile_num + self.fused_input_size + self.fused_output_size

        if mode == 1:
            p2 = self.resource.access_cost[2]
            p1 = self.resource.access_cost[1]

            q0, q1, q2 = p1[0], p1[1], p2[2] + p1[2]

            f_args = (q0, q1, q2, b, h_m)
            fun = self.fun(f_args)
            c_args = (b, h_m, self.idx_dict)
            con = self.con(c_args)
            x0 = [1 for _ in range(len(set(self.idx_dict.values())) + 1)]
            if b > 1:
                x0[0] = b
            else:
                x0[0] = h
            for idx in self.idx_dict:
                if idx < len(self.dag_vertex_list):
                    layer = self.network[self.dag_vertex_list[idx]]
                    if isinstance(layer, LocalRegionLayer):
                        continue
                    loop_lower_bound = self.loop_lower_bound(layer)
                    x0[self.idx_dict[idx]] = loop_lower_bound.k

            x0 = np.asarray(x0)
            res = minimize(fun, x0, method='COBYLA', constraints=con)

            if res.success:
                if b > 1:
                    b = math.ceil(res.x[0])
                    h = h_m
                else:
                    b = 1
                    h = math.ceil(res.x[0])
                H = [None for _ in self.dag_vertex_list]
                for l in reversed(self.dag_vertex_list):
                    idx = self.dag_vertex_dict[l]
                    layer = self.network[l]
                    if l in self.lasts:
                        H[idx] = int(self.scale[idx].s_h * h)
                        continue

                    h_tmp = None
                    for dst_idx in self.dag_next_dict[idx]:
                        dst = self.dag_vertex_list[dst_idx]
                        dst_layer = self.network[dst]
                        dst_h = H[dst_idx]
                        assert dst_h is not None
                        if isinstance(dst_layer, LocalRegionLayer):
                            hreg = dst_layer.hreg
                        else:
                            hreg = dst_layer.hfil
                        if h_tmp is None:
                            h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                        else:
                            if (dst_h - 1) * dst_layer.hstd + hreg > h_tmp:
                                h_tmp = min((dst_h - 1) * dst_layer.hstd + hreg, layer.hofm)
                    H[idx] = math.floor(h_tmp)

                for l in self.dag_vertex_list:
                    idx = self.dag_vertex_dict[l]
                    layer = self.network[l]
                    if isinstance(layer, LocalRegionLayer):
                        continue

                    if idx in self.idx_dict:
                        k = res.x[self.idx_dict[idx]]
                    else:
                        k = layer.nofm

                    if self.dag_prev_dict[idx] and list(self.dag_prev_dict[idx])[0] in self.idx_dict:
                        c = self.idx_dict[list(self.dag_prev_dict[idx])[0]]
                    else:
                        c = layer.nifm
                    self.loop_block[idx] = \
                        [layer.wfil, layer.hfil, math.ceil(c), layer.wofm, H[idx], math.ceil(k), b]

                    self.loop_order[idx] = \
                        schedule_generator.loop_order_generator(layer, self.loop_block[idx], irrelevant)

            q = self.fused_weight_size * self.network.input_layer().nimg * h_m * q2 / (b * h) \
                + self.fused_input_size * p2[0] + self.fused_output_size * p2[1]
            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                if isinstance(layer, LocalRegionLayer):
                    continue
                q += (q1 * layer.nifm * layer.total_ofmap_size / self.loop_block[idx][2])
                q += (q0 * layer.nofm * layer.total_ifmap_size / self.loop_block[idx][5])
            self.q = q
            self.tile_num = math.ceil(h_m * self.network.input_layer().nimg / (b * h))

        return True

    def _init_alternate_pair(self, mode):
        if self.d_fusion or self.z_fusion:
            return self._init_alternate_pair_others(mode)
        else:
            return self._init_alternate_pair_optimus()

    def fun(self, args):

        print('func')
        q0, q1, q2, b, h_m = args
        expr = ''

        fidx = 1
        idx_dict = dict()
        if b > 1:
            p2 = q2 * self.fused_weight_size * self.network.input_layer().nimg
            expr += '+ {p2} / x[0] '.format(p2=p2)
        else:
            p2 = q2 * self.fused_weight_size * self.network.input_layer().nimg * h_m
            expr += '+ {p2} / x[0] '.format(p2=p2)
        print(f'list:{self.dag_vertex_list}')
        for l in self.dag_vertex_list:
            idx = self.dag_vertex_dict[l]
            layer = self.network[l]
            print(f'layer({idx}):{l}')

            if isinstance(layer, ConvLayer):
                assert len(self.dag_prev_dict[idx]) <= 1
                p0 = q0 * layer.total_ifmap_size * layer.nofm
                p1 = q1 * layer.total_ofmap_size * layer.nifm

                k = True if self.is_full_buffer[idx] else False
                c = False
                for pre in self.dag_prev_dict[idx]:
                    if self.is_full_buffer[pre]:
                        c = True
                if not k and not c:
                    c = True

                print(f'k:{k},c:{c}')

                if not k:
                    if idx in idx_dict:
                        cur_fidx = idx_dict[idx]
                        print(f'cur_fidx:{cur_fidx},{self.dag_vertex_list[cur_fidx]}')
                    else:
                        cur_fidx = fidx
                        idx_dict[idx] = fidx
                        fidx += 1
                        #print(f'cur_fidx:{cur_fidx},{self.dag_vertex_list[cur_fidx]}')
                    expr += '+ {p0} / x[{idx}] '.format(p0=p0, idx=cur_fidx)
                print(f'idx_dict:{idx_dict}')

                if not c:
                    if len(self.dag_prev_dict[idx]) == 1:
                        pidx = list(self.dag_prev_dict[idx])[0]
                        print(f'pidx:{pidx},{self.dag_vertex_list[pidx]}')
                        if pidx in idx_dict:
                            cur_fidx = idx_dict[pidx]
                            print(f'cur_fidx:{cur_fidx}')
                        else:
                            cy_idx = pidx
                            print(f'cy_idx:{cy_idx}')
                            while len(self.dag_prev_dict[cy_idx]) == 1:
                                if isinstance(self.network[self.dag_vertex_list[cy_idx]], ConvLayer):
                                    break
                                cy_idx = list(self.dag_prev_dict[cy_idx])[0]
                                print(f'cy_idx:{cy_idx}')
                            if len(self.dag_prev_dict[cy_idx]) == 1:
                                if cy_idx in idx_dict:
                                    cur_fidx = idx_dict[cy_idx]
                                    idx_dict[pidx] = cur_fidx
                                    print(f'idx_dict2:{idx_dict}')
                                else:
                                    cur_fidx = fidx
                                    idx_dict[cy_idx] = cur_fidx
                                    idx_dict[pidx] = cur_fidx
                                    fidx += 1
                                    print(f'idx_dict3:{idx_dict}')
                            elif len(self.dag_prev_dict[cy_idx]) == 0:
                                continue

                            else:
                                cur_fidx = fidx
                                idx_dict[pidx] = cur_fidx
                                fidx += 1

                    else:
                        continue

                    expr += '+ {p1} / x[{idx}] '.format(p1=p1, idx=cur_fidx)

        self.idx_dict = idx_dict
        expr = expr[1:]
        v = lambda x: eval(expr)
        return v

    def con(self, args):
        b, h_m, idx_dict = args

        ineq_cons = []
        if b > 1:
            ineq_cons.append('x[0] - 1')
            ineq_cons.append('-x[0] + {nimg}'.format(nimg=self.network.input_layer().nimg))
        else:
            ineq_cons.append('x[0] - 1')
            ineq_cons.append('-x[0] + {hh}'.format(hh=h_m))

        ext_inputs = set(self.ext_inputs_dict.keys())
        ss = ''
        layer_oc = {}
        ext_input_ic = {}
        kernel_size = {}
        layer_ofmap_size = {}
        layer_fmap_size = {}
        buffer_const = 'max(['

        for l in self.dag_vertex_list:
            idx = self.dag_vertex_dict[l]
            layer = self.network[l]
            sca = self.scale[idx]
            minsize = self.minSize[idx]
            print(f'minsize:{minsize}')
            loop_lower_bound = self.loop_lower_bound(layer)

            ic = str(0)
            ih = str(0)
            if l in self.firsts:
                if not self.is_full_buffer[idx]:
                    for src in self.network.prevs(l):
                        if src in ext_inputs:
                            if src is None:
                                src_layer = self.network.input_layer()
                            else:
                                src_layer = self.network[src]
                            ic = str(src_layer.nofm)
                            m_h = min((minsize.h - 1) * layer.hstd + layer.hfil, src_layer.hofm)
                            s_h = sca.s_h * layer.hstd
                            if b > 1:
                                ih = str(layer.hifm)
                                ss += '+x[0]*{hh}*{w}*{k}'\
                                    .format(hh=layer.hifm, w=layer.wifm, k=src_layer.nofm)
                            else:
                                oh = '({m_h}-{s_h}+{s_h}*x[0])'.format(m_h=m_h, s_h=s_h)
                                ih = 'min({oh}-1)*{hstd} + {hfil},{hifm})'.format(oh=oh, hstd=layer.hstd, hfil=layer.hfil, hifm=layer.hifm)
                                ss += '+({m_h}-{s_h}+{s_h}*x[0])*{w}*{k}' \
                                    .format(m_h=m_h, s_h=s_h, w=layer.wifm, k=src_layer.nofm)

                            layer_oc[src_layer] = str(src_layer.nofm)
                            ext_inputs.remove(src)

                print(f'layer_oc:{layer_oc}')

                if isinstance(layer, LocalRegionLayer) and self.is_full_buffer[idx]:
                    if b > 1:
                        ss += '+x[0]*{hh}*{w}*{k}'.format(hh=layer.hofm, w=layer.wofm, k=layer.nofm)
                        if layer in layer_ofmap_size:
                            layer_ofmap_size[layer] += '+x[0]*{hh}*{w}*{k}'.format(hh=layer.hofm, w=layer.wofm, k=layer.nofm)
                        else:
                            layer_ofmap_size[layer] = '+x[0]*{hh}*{w}*{k}'.format(hh=layer.hofm, w=layer.wofm, k=layer.nofm)
                        if layer in layer_fmap_size:
                            layer_fmap_size[layer] += '+x[0]*{hh}*{w}*{k}'.format(hh=layer.hofm, w=layer.wofm, k=layer.nofm)
                        else:
                            layer_fmap_size[layer] = '+x[0]*{hh}*{w}*{k}'.format(hh=layer.hofm, w=layer.wofm, k=layer.nofm)
                    else:
                        ss += '+({m_h}-{s_h}+{s_h}*x[0])*{w}*{k}'\
                            .format(m_h=minsize.h, s_h=sca.s_h, w=layer.wofm, k=layer.nofm)

                        if layer in layer_ofmap_size:
                            layer_ofmap_size[layer] += '+({m_h}-{s_h}+{s_h}*x[0])*{w}*{k}'\
                                .format(m_h=minsize.h, s_h=sca.s_h, w=layer.wofm, k=layer.nofm)
                        else:
                            layer_ofmap_size[layer] = '+({m_h}-{s_h}+{s_h}*x[0])*{w}*{k}'\
                                .format(m_h=minsize.h, s_h=sca.s_h, w=layer.wofm, k=layer.nofm)
                        if layer in layer_fmap_size:
                            layer_fmap_size[layer] += '+({m_h}-{s_h}+{s_h}*x[0])*{w}*{k}'\
                                .format(m_h=minsize.h, s_h=sca.s_h, w=layer.wofm, k=layer.nofm)
                        else:
                            layer_fmap_size[layer] = '+({m_h}-{s_h}+{s_h}*x[0])*{w}*{k}'\
                                .format(m_h=minsize.h, s_h=sca.s_h, w=layer.wofm, k=layer.nofm)

                    layer_oc[layer] = str(layer.nofm)
            #else:
            #    kernel_size[idx+1] += 'x[{idx}]'.format(idx=idx)


            if isinstance(layer, ConvLayer):
                print(f'con layer:{self.dag_vertex_list[idx]}')
                assert len(self.dag_prev_dict[idx]) <= 1
                ocfull = True if self.is_full_buffer[idx] else False
                icfull = False
                for pre in self.dag_prev_dict[idx]:
                    if self.is_full_buffer[pre]:
                        icfull = True
                if not ocfull and not icfull:
                    icfull = True

                if not l in self.firsts:
                    if not icfull:
                        pidx = list(self.dag_prev_dict[idx])[0]
                        while not pidx in idx_dict:
                            if len(self.dag_prev_dict[pidx]) == 1:
                                pidx = list(self.dag_prev_dict[pidx])[0]
                            else:
                                assert pidx in idx_dict
                        print(f'pidx:{pidx},{idx_dict}')
                        ic = 'x[{pidx}]'.format(pidx=idx_dict[pidx])
                    else:
                        ic = str(self.network[list(self.network.prevs(l))[0]].nofm)
                else:
                    src = list(self.network.prevs(l))[0]
                    if src is None:
                        src_layer = self.network.input_layer()
                    else:
                        src_layer = self.network[src]
                    ic = str(src_layer.nofm)

                print(f'ic:{ic}')

                if self.is_full_buffer[idx]:
                    if b > 1:
                        oh = str(layer.hofm)
                        ih = str(layer.hifm)
                        ss += '+x[0]*{hh}*{w}*{k}' \
                            .format(hh=layer.hofm, w=layer.wofm, k=layer.nofm)
                    else:
                        oh = '({m_h}-{s_h}+{s_h}*x[0])'.format(m_h=minsize.h, s_h=sca.s_h)
                        ih = 'min((({oh})-1)*{hstd} + {hfil},{hifm})'.format(oh=oh, hstd=layer.hstd, hfil=layer.hfil, hifm=layer.hifm)
                        ss += '+({m_h}-{s_h}+{s_h}*x[0])*{w}*{k}' \
                            .format(m_h=minsize.h, s_h=sca.s_h, w=layer.wofm, k=layer.nofm)
                    layer_oc[layer] = str(layer.nofm)
                    kernel_size[layer] = '{hfil}*{wfil}*({ic})*{oc}'.format(hfil=layer.hfil, wfil=layer.wfil, ic=ic, oc=layer.nofm)
                    layer_ofmap_size[layer] = '{oh}*{ow}*{oc}'.format(oh=oh, ow=layer.wofm, oc=layer.nofm)
                    layer_fmap_size[layer] = '(({ih})*{iw}*({ic}))+({ofmap}*(1 if {ic}=={nifm} else 32/{precision}))'.format(ih=ih, iw=layer.wifm, ic=ic, ofmap=layer_ofmap_size[layer], nifm=layer.nifm, precision=self.resource.precision)

                else:
                    cur_fidx = idx_dict[idx]
                    if b > 1:
                        oh = str(layer.hofm)
                        ih = str(layer.hifm)
                        ss += '+x[0]*{hh}*{w}*x[{idx}]' \
                            .format(hh=layer.hofm, w=layer.wofm, idx=cur_fidx)
                    else:
                        oh = '({m_h}-{s_h}+{s_h}*x[0])'.format(m_h=minsize.h, s_h=sca.s_h)
                        ih = 'min((({oh})-1)*{hstd} + {hfil},{hifm})'.format(oh=oh, hstd=layer.hstd, hfil=layer.hfil, hifm=layer.hifm)
                        ss += '+({m_h}-{s_h}+{s_h}*x[0])*{w}*x[{idx}]' \
                            .format(m_h=minsize.h, s_h=sca.s_h, w=layer.wofm, idx=cur_fidx)

                    layer_oc[layer] = 'x[{idx}]'.format(idx=cur_fidx)

                    ineq_cons.append('x[{idx}] - {k}'.format(idx=cur_fidx, k=loop_lower_bound.k))
                    ineq_cons.append('-x[{idx}] + {nofm}'.format(idx=cur_fidx, nofm=layer.nofm))
                    kernel_size[layer] = '{hfil}*{wfil}*({ic})*x[{idx}]'.format(hfil=layer.hfil, wfil=layer.wfil, ic=ic, idx=cur_fidx)
                    layer_ofmap_size[layer] = '{oh}*{ow}*x[{idx}]'.format(oh=oh, ow=layer.wofm, idx=cur_fidx)
                    layer_fmap_size[layer] = '(({ih})*{iw}*({ic}))+({ofmap}*(1 if {ic}=={nifm} else 32/{precision}))'.format(ih=ih, iw=layer.wifm, ic=ic, ofmap=layer_ofmap_size[layer], nifm=layer.nifm, precision=self.resource.precision)
                    #layer_fmap_size[layer] = '(({ih})*{iw}*({ic}))+({ofmap})'.format(ih=ih, iw=layer.wifm, ic=ic, ofmap=layer_ofmap_size[layer])

                #layer_fmap_size[layer] = layer_fmap_size[layer][1:]+'+{}'.format(layer_ofmap_size[src_layer][1:])

                buffer_const += '({fmap_size})+({kernel_size}),'.format(fmap_size=layer_fmap_size[layer], kernel_size=kernel_size[layer])
        buffer_const = buffer_const[:-1]+'])'
        print(f'buffer_const:{buffer_const}')

        for src in ext_inputs:
            if src is None:
                src_layer = self.network.input_layer()
            else:
                src_layer = self.network[src]
            loop_lower_bound = self.loop_lower_bound(src_layer)
            pidx = self.ext_inputs_idx[src]
            if pidx in idx_dict:
                cur_fidx = idx_dict[pidx]
                ineq_cons.append('x[{pidx}] - {k}'.format(pidx=cur_fidx, k=loop_lower_bound.k))
                ineq_cons.append('-x[{pidx}] + {nofm}'.format(pidx=cur_fidx, nofm=src_layer.nofm))

        s = self.resource.buffer(1).capacity
        #ss = '-(' + ss[1:] + ')+{}'.format(s)
        ss = '-(' + buffer_const + ')+{}'.format(s)
        ineq_cons.append(ss)
        cons = ()
        for ineq in ineq_cons:
            print(f'ineq:{ineq}')
            cons += ({'type': 'ineq', 'fun': lambda x, ineq=ineq: eval(ineq)}, )

        cons_res = copy.copy(cons)
        print(f'cons_res:{cons_res}')
        return cons_res

    def _alternate(self):

        level = self.resource.buffer_levels() - 2
        s = self.resource.buffer(level).capacity * self.resource.paras[level].count

        min_feature_footprint_t = s - 0.0000001
        min_required_mem_size_t = s - 0.0000001
        add_one_line_footprint_t = float('inf')

        is_full_buffer_t = None
        print('\n||alternate')
        print('firsts:',self.firsts)
        print('lasts:',self.lasts)
        for start in [True, False]:
            print('start: ',start)
            is_full_buffer = [None for _ in range(len(self.dag_vertex_list))]
            min_feature_footprint = 0
            add_one_line_footprint = 0
            min_required_mem_size = 0
            layer_ofmap_size = [0 for _ in range(len(self.dag_vertex_list))]
            layer_fmap_size = [0 for _ in range(len(self.dag_vertex_list))]
            layer_kernel_size = [0 for _ in range(len(self.dag_vertex_list))]
            ext_inputs = set(self.ext_inputs_dict.keys())
            for l in self.dag_vertex_list:
                idx = self.dag_vertex_dict[l]
                layer = self.network[l]
                sca = self.scale[idx]
                minsize = self.minSize[idx]
                print(str(idx),':',l,':',layer )
                if l in self.firsts:
                    if self.womincost:
                        is_full_buffer[idx] = True
                    else:
                        if start:
                            is_full_buffer[idx] = True
                        else:
                            is_full_buffer[idx] = False

                    #if not is_full_buffer[idx]:
                    for src in self.network.prevs(l):
                        if src in ext_inputs:
                            if src is None:
                                src_layer = self.network.input_layer()
                            else:
                                src_layer = self.network[src]

                            print('before min_feature_footprint:',
                                    str(min_feature_footprint))
                            print('before add_one_line_footprint:',
                                    str(add_one_line_footprint))
                            min_feature_footprint += \
                                (src_layer.nofm
                                 * min(((minsize.h - 1) * layer.hstd + layer.hfil), src_layer.hofm)
                                 * layer.wifm)
                            add_one_line_footprint += \
                                (src_layer.nofm
                                 * sca.s_h * layer.hstd
                                 * layer.wifm)
                            layer_fmap_size[idx] += \
                                (src_layer.nofm
                                 * min(((minsize.h - 1) * layer.hstd + layer.hfil), src_layer.hofm)
                                 * layer.wifm)
                            print('  src_layer:',src,':',src_layer)
                            print('  min_feature_footprint1+',
                                  str((src_layer.nofm
                                 * min(((minsize.h - 1) * layer.hstd + layer.hfil), src_layer.hofm)
                                 * layer.wifm)),'=',min_feature_footprint)
                            print('  (src_layer.nofm(',str(src_layer.nofm),
                                  ')*h(',
                                  str(min(((minsize.h - 1) * layer.hstd + layer.hfil), src_layer.hofm)),
                                  ')*layer.wifm(',
                                  str(layer.wifm),'))')

                            print('  add_one_line_footprint1+',
                                  str((src_layer.nofm
                                 * sca.s_h * layer.hstd
                                 * layer.wifm)),'=',add_one_line_footprint)
                            print('  (src_layer.nofm(',str(src_layer.nofm),
                                  ')*sca.s_h(',str(sca.s_h),
                                  ')*layer.hstd(',str(layer.hstd),
                                  ')*layer.wifm(',
                                  str(layer.wifm),'))')

                            ext_inputs.remove(src)
                    if isinstance(layer, LocalRegionLayer):
                        if is_full_buffer[idx]:
                            print('before min_feature_footprint:',
                                    str(min_feature_footprint))
                            print('before add_one_line_footprint:',
                                    str(add_one_line_footprint))
                            min_feature_footprint += layer.nofm * minsize.h * layer.wofm
                            layer_ofmap_size[idx] = layer.nofm * minsize.h * layer.wofm
                            add_one_line_footprint += layer.nofm * sca.s_h * layer.wofm
                            layer_fmap_size[idx] += layer.nofm * minsize.h * layer.wofm
                            print('  min_feature_footprint2+',
                                  'layer.norm(',str(layer.nofm),
                                  ') * minsize.h(', str(minsize.h),
                                  ') * layer.wofm(',str(layer.wofm),'):',
                                  str(layer.nofm*minsize.h*layer.wofm),
                                  '=',min_feature_footprint)
                            print('  add_one_line_footprint2+',
                                  'layer.norm(',str(layer.nofm),
                                  ') * sca.s_h(', str(sca.s_h),
                                  ') * layer.wofm(',str(layer.wofm),'):',
                                  str(layer.nofm*sca.s_h*layer.wofm),
                                  '=',add_one_line_footprint)
                        else:
                            k = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nofm)
                            print('before min_feature_footprint:',
                                    str(min_feature_footprint))
                            print('before add_one_line_footprint:',
                                    str(add_one_line_footprint))
                            min_feature_footprint += k * minsize.h * layer.wofm
                            layer_ofmap_size[idx] = k * minsize.h * layer.wofm
                            add_one_line_footprint += k * sca.s_h * layer.wofm
                            layer_fmap_size[idx] += k * minsize.h * layer.wofm
                            print('  min_feature_footprint2.1+',
                                  'k(',str(k),
                                  ') * minsize.h(', str(minsize.h),
                                  ') * layer.wofm(',str(layer.wofm),'):',
                                  str(k*minsize.h*layer.wofm),
                                  '=',min_feature_footprint)
                            print('  add_one_line_footprint2.1+',
                                  'k',str(k),
                                  ') * sca.s_h(', str(sca.s_h),
                                  ') * layer.wofm(',str(layer.wofm),'):',
                                  str(k*sca.s_h*layer.wofm),
                                  '=',add_one_line_footprint)


                for src_idx in self.dag_prev_dict[idx]:
                    assert is_full_buffer[src_idx] is not None
                    if self.womincost:
                        is_full_buffer[idx] = True
                    else:
                        if isinstance(layer, LocalRegionLayer):
                            if is_full_buffer[idx] is None:
                                is_full_buffer[idx] = is_full_buffer[src_idx]
                            else:
                                is_full_buffer[idx] \
                                    = is_full_buffer[idx] or is_full_buffer[src_idx]
                        else:
                            if not is_full_buffer[src_idx]:
                                is_full_buffer[idx] = True
                            else:
                                is_full_buffer[idx] = False
                    layer_fmap_size[idx] += layer_ofmap_size[src_idx]

                #if isinstance(layer, ConvLayer):
                print('  is_full_buffer:',is_full_buffer[idx])
                if is_full_buffer[idx]:
                    print('before min_feature_footprint:',
                            str(min_feature_footprint))
                    print('before add_one_line_footprint:',
                            str(add_one_line_footprint))
                    min_feature_footprint += layer.nofm * minsize.h * layer.wofm
                    layer_ofmap_size[idx] = layer.nofm * minsize.h * layer.wofm
                    add_one_line_footprint += layer.nofm * sca.s_h * layer.wofm
                    layer_fmap_size[idx] += layer.nofm * minsize.h * layer.wofm
                    print('  min_feature_footprint3+',
                          'layer.nofm(',str(layer.nofm),
                          ') * minsize.h(', str(minsize.h),
                          ') * layer.wofm(',str(layer.wofm),'):',
                          str(layer.nofm*minsize.h*layer.wofm),
                          '=',min_feature_footprint)
                    print('  add_one_line_footprint3+',
                          'layer.nofm(',str(layer.nofm),
                          ') * sca.s_h(', str(sca.s_h),
                          ') * layer.wofm(',str(layer.wofm),'):',
                          str(layer.nofm*sca.s_h*layer.wofm),
                          '=',add_one_line_footprint)

                else:
                    loop_lower_bound = self.loop_lower_bound(layer)
                    k = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nofm)
                    print('  k:',k)
                    print('  max(loop_lower_bound.k(',str(loop_lower_bound.k),
                          '),loop_lower_bound.c(',str(loop_lower_bound.c),'))=',
                          str(max(loop_lower_bound.k,loop_lower_bound.c)))
                    print('  min(',str(max(loop_lower_bound.k,loop_lower_bound.c)),
                          ',layer.nofm(',str(layer.nofm),'))=',
                          str(min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nofm)))


                    print('before min_feature_footprint:',
                            str(min_feature_footprint))
                    print('before add_one_line_footprint:',
                            str(add_one_line_footprint))
                    min_feature_footprint += k * minsize.h * layer.wofm
                    layer_ofmap_size[idx] = k * minsize.h * layer.wofm
                    add_one_line_footprint += k * sca.s_h * layer.wofm
                    layer_fmap_size[idx] += k * minsize.h * layer.wofm
                    print('  min_feature_footprint4+',
                          'k(',str(k),
                          ') * minsize.h(', str(minsize.h),
                          ') * layer.wofm(',str(layer.wofm),'):',
                          str(k*minsize.h*layer.wofm),
                          '=',min_feature_footprint)
                    print('  add_one_line_footprint4+',
                          'k(',str(k),
                          ') * sca.s_h(', str(sca.s_h),
                          ') * layer.wofm(',str(layer.wofm),'):',
                          str(k*sca.s_h*layer.wofm),
                          '=',add_one_line_footprint)

            max_kernel_size = 0
            print('calc kernel_size')
            for l in self.dag_vertex_list:
                layer = self.network[l]
                idx = self.dag_vertex_dict[l]
                print(str(idx),':',l,':',layer )
                if isinstance(layer, ConvLayer):
                    kernel_size = 0
                    ki = layer.nifm;
                    for src_idx in self.dag_prev_dict[idx]:
                        if is_full_buffer[src_idx]:
                            ki = layer.nifm 
                        else:
                            ki = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nifm)

                    if is_full_buffer[idx]:
                        kernel_size = layer.nofm * ki * layer.wfil * layer.hfil
                    else:
                        k = min(max(loop_lower_bound.k, loop_lower_bound.c), layer.nofm)
                        kernel_size = k * ki * layer.wfil * layer.hfil
                    layer_kernel_size[idx] = kernel_size
                    print('kernel_size:',kernel_size)
                    if (kernel_size > max_kernel_size):
                        max_kernel_size = kernel_size;

            print(f'layer_fmap_size:{layer_fmap_size}')
            print(f'layer_kernel_size:{layer_kernel_size}')
            min_required_mem_size = max([sum(x) for x in zip(layer_fmap_size,layer_kernel_size)])
            print(f'min_required_mem_size:{min_required_mem_size}')


            print('max_kernel_size:',max_kernel_size)
            #min_feature_footprint += max_kernel_size;
            print(f's:{s}, min_feature_footprint:{min_feature_footprint}')
            print(f'one_line:{add_one_line_footprint}')
            if (s - min_feature_footprint) > 0 \
                    and (add_one_line_footprint / (s - min_feature_footprint)) \
                    < (add_one_line_footprint_t / (s - min_feature_footprint_t)):
                min_feature_footprint_t = min_feature_footprint
                is_full_buffer_t = is_full_buffer
                add_one_line_footprint_t = add_one_line_footprint
                min_required_mem_size_t = min_required_mem_size
        return min_feature_footprint_t, is_full_buffer_t, add_one_line_footprint_t, min_required_mem_size_t

