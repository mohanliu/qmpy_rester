#import networkx as nx
from scipy.spatial import ConvexHull
import matplotlib.pylab as plt

from .phase import *
from .equilibrium import Equilibrium
from ..utils import *

class PhaseSpaceError(Exception):
    pass

class Heap(dict):
    def add(self, seq):
        if len(seq) == 1:
            self[seq[0]] = Heap()
            return 
        seq = sorted(seq)
        e0 = seq[0]
        if e0 in self:
            self[e0].add(seq[1:])
        else:
            self[e0] = Heap()
            self[e0].add(seq[1:])

    @property
    def sequences(self):
        seqs = []
        for k, v in self.items():
            if not v:
                seqs.append([k])
            else:
                for v2 in v.sequences:
                    seqs.append([k] + v2)
        return seqs

class PhaseSpace(object):
    """
    A PhaseSpace object represents, naturally, a region of phase space.

    The most fundamental property of a PhaseSpace is its bounds,
    which are given as a hyphen-delimited list of compositions. These represent
    the extent of the phase space, and determine which phases are within the
    space.

    Next, a PhaseSpace has an attribute, data, which is a PhaseData object,
    and is a container for Phase objects, which are used when performing
    thermodynamic analysis on this space.

    The majority of attributes are lazy, that is, they are only computed when
    they are requested, and how to get them (of which there are often several
    ways) is decided based on the size and shape of the phase space.

    """

    def __init__(self, bounds, mus=None, data=None, **kwargs):
        """
        Arguments:
            bounds:
                Sequence of compositions. Can be comma-delimited ("Fe,Ni,O"),
                an actual list (['Fe', 'Ni', 'O']) or any other python
                sequence. The compositions need not be elements, if you want to
                take a slice through the Fe-Ni-O phase diagram between Fe3O4
                and NiO, just do "Fe3O4-NiO".

        Keyword Arguments
            mus:
                define a dictionary of chemical potentials. Will adjust all
                calculated formation energies accordingly.

            data:
                If supplied with a PhaseData instance, it will be used
                instead of loading from the OQMD. Can be used to significantly
                reduce the amount of time spent querying the database when looping
                through many PhaseSpaces.

        Examples::

            >>> ps = PhaseSpace('Fe-Li-O', load="legacy.dat")
            >>> ps2 = PhaseSpace(['Fe','Li','O'], data=ps.data)
            >>> ps = PhaseSpace(set(['Li', 'Ni', 'O']))
            >>> ps = PhaseSpace('Li2O-Fe2O3')

        """
        self.clear_all()
        self.set_mus(mus)
        self.set_bounds(bounds)
        if data is None:
            self.data = PhaseData()
            if bounds:
                self.load(**kwargs)
        else:
            self.data = data.get_phase_data(self.space)

    def __repr__(self):
        if self.bounds is None:
            return '<unbounded PhaseSpace>'
        names = [ format_comp(reduce_comp(b)) for b in self.bounds ]
        bounds = '-'.join(names)
        if self.mus:
            bounds += ' ' + format_mus(self.mus)
        return '<PhaseSpace bound by %s>' % bounds

    def __getitem__(self, i):
        return self.phases[i]

    def __len__(self):
        return len(self.phases)

    def set_bounds(self, bounds):
        bounds = parse_space(bounds)
        if bounds is None:
            self.bounds = None
            return 

        elements = sorted(set.union(*[ set(b.keys()) for b in bounds ]))
        basis = []
        for b in bounds:
            basis.append([ b.get(k, 0) for k in elements])

        self.bounds = bounds
        self.basis = np.array(basis)

    def infer_formation_energies(self):
        mus = {}
        for elt in self.space:
            if elt in self.phase_dict:
                mus[elt] = self.phase_dict[elt].energy
            else:
                mus[elt] = 0.0

        for phase in self.phases:
            for elt in self.space:
                phase.energy -= phase.unit_comp.get(elt, 0)*mus[elt]

    def set_mus(self, mus):
        self.mus = {}
        if mus is None:
            return
        elif isinstance(mus, str):
            mus = mus.replace(',', ' ')
            for mu in mus.split():
                self.mus.update(parse_mu(mu))
        elif isinstance(mus, dict):
            self.mus = mus

    def get_subspace(self, space):
        data = self.data.get_phase_data(space)
        return PhaseSpace(space, data=data)

    _phases = None
    @property
    def phases(self):
        if self._phases:
            return self._phases
        phases = [ p for p in self.data.phases if self.in_space(p) and p.use ]
        self._phases = phases
        return self._phases

    @phases.setter
    def phases(self, phases):
        self.clear_all()
        self.data = PhaseData()
        self.data.phases = phases

    _phase_dict = None
    @property
    def phase_dict(self):
        if self._phase_dict:
            return self._phase_dict
        phase_dict = dict([ (k, p) for k, p in self.data.phase_dict.items()
                if p.use and self.in_space(p) ])
        self._phase_dict = phase_dict
        return self._phase_dict

    @phase_dict.setter
    def phase_dict(self, phase_dict):
        self.clear_all()
        self.data = PhaseData()
        self.data.phases = phase_dict.values()

    def phase_energy(self, p):
        dE = sum([self.mus.get(k, 0)*v for k,v in p.unit_comp.items()])
        N = sum(v for k,v in p.unit_comp.items() if k in self.bound_space)
        if N == 0:
            N = 1
        return (p.energy - dE)/N

    def phase_comp(self, p):
        comp = dict((k,v) for k,v in p.comp.items() 
                 if k in self.bound_elements)
        return unit_comp(comp)

    def clear_data(self):
        """
        Clears all phase data.
        """
        self._phases = None
        self._phase_dict = None

    def clear_analysis(self):
        """
        Clears all calculated results.
        """
        self._stable = None
        self._tie_lines = None
        self._hull = None
        self._spaces = None
        self._dual_spaces = None
        self._cliques = None
        self._graph = None

    def clear_all(self):
        """
        Clears input data and analyzed results. 
        Same as:
        >>> PhaseData.clear_data() 
        >>> PhaseData.clear_analysis()
        """
        self.clear_data()
        self.clear_analysis()

    @property
    def comp_dimension(self):
        """
        Compositional dimension of the region of phase space.

        Examples::

            >>> s = PhaseSpace('Fe-Li-O')
            >>> s.comp_dimension
            2
            >>> s = PhaseSpace('FeO-Ni2O-CoO-Ti3O4')
            >>> s.comp_dimension
            3

        """
        return len(self.bounds) - 1

    @property
    def shape(self):
        """
        (# of compositional dimensions, # of chemical potential dimensions)
        The shape attribute of the PhaseSpace determines what type of phase
        diagram will be drawn.

        Examples::

            >>> s = PhaseSpace('Fe-Li', 'O=-1.2')
            >>> s.shape
            (1, 0)
            >>> s = PhaseSpace('Fe-Li', 'O=0:-5')
            >>> s.shape
            (1, 1)
            >>> s = PhaseSpace('Fe-Li-P', 'O=0:-5')
            >>> s.shape
            (2,1)
            >>> s = PhaseSpace('Fe', 'O=0:-5')
            >>> s.shape
            (0, 1)

        """
        return (self.comp_dimension, self.chempot_dimension)

    @property
    def bound_space(self):
        """
        Set of elements _of fixed composition_ in the PhaseSpace.

        Examples::

            >>> s = PhaseSpace('Fe-Li', 'O=-1.4')
            >>> s.bound_space
            set(['Fe', 'Li'])

        """
        if self.bounds is None:
            return set()
        return set.union(*[ set(b.keys()) for b in self.bounds ])

    @property
    def bound_elements(self):
        """
        Alphabetically ordered list of elements with constrained composition.
        """
        return sorted(self.bound_space)

    @property
    def space(self):
        """
        Set of elements present in the PhaseSpace.

        Examples::

            >>> s = PhaseSpace('Pb-Te-Se')
            >>> s.space
            set(['Pb', 'Te', 'Se'])
            >>> s = PhaseSpace('PbTe-Na-PbSe')
            >>> s.space
            set(['Pb', 'Te', 'Na', 'Se'])

        """
        return self.bound_space | set(self.mus.keys())

    @property
    def elements(self):
        """
        Alphabetically ordered list of elements present in the PhaseSpace.
        """
        return sorted(self.space)

    def coord(self, composition, tol=1e-4):
        """Returns the barycentric coordinate of a composition, relative to the
        bounds of the PhaseSpace. If the object isn't within the bounds, raises
        a PhaseSpaceError.

        Examples::

            >>> space = PhaseSpace('Fe-Li-O')
            >>> space.coord({'Fe':1, 'Li':1, 'O':2})
            array([ 0.25,  0.25,  0.5 ])
            >>> space = PhaseSpace('Fe2O3-Li2O')
            >>> space.coord('Li5FeO4')
            array([ 0.25,  0.75])

        """
        if isinstance(composition, Phase):
            composition = composition.comp
        elif isinstance(composition, str):
            composition = parse_comp(composition)

        composition = defaultdict(float, composition)
        if self.bounds is None:
            return np.array([ composition[k] for k in self.bound_elements ])

        bcomp = dict((k,v) for k,v in composition.items() if k in
                self.bound_space)
        composition = unit_comp(bcomp)
        cvec = np.array([ composition.get(k, 0) for k in self.bound_elements ])
        coord = np.linalg.lstsq(self.basis.T, cvec, rcond=None)[0]
        if abs(sum(coord) - 1) > 1e-3 or any(c < -1e-3 for c in coord):
            raise PhaseSpaceError
        return coord

    def comp(self, coord):
        """
        Returns the composition of a coordinate in phase space.

        Examples::

            >>> space = PhaseSpace('Fe-Li-O')
            >>> space.comp([0.2, 0.2, 0.6])
            {'Fe': 0.2, 'O': 0.6, 'Li': 0.2}

        """
        if self.bounds is None:
            return defaultdict(float, zip(self.elements, coord))
        if len(coord) != len(self.bounds):
            raise PhaseSpaceError
        if len(coord) != len(self.bounds):
            raise ValueError("Dimensions of coordinate must match PhaseSpace")

        tot = sum(coord)
        coord = [ c/float(tot) for c in coord ]
        comp = defaultdict(float)
        for b, x in zip(self.bounds, coord):
            for elt, val in b.items():
                comp[elt] += val*x
        return dict( (k,v) for k,v in comp.items() if v > 1e-4 )

    _spaces = None
    @property
    def spaces(self):
        """
        List of lists of elements, such that every phase in self.phases
        is contained in at least one set, and no set is a subset of
        any other. This corresponds to the smallest subset of spaces that must
        be analyzed to determine the stability of every phase in your dataset.

        Examples::

            >>> pa, pb, pc = Phase('A', 0), Phase('B', 0), Phase('C', 0)
            >>> p1 = Phase('AB2', -1)
            >>> p2 = Phase('B3C', -4)
            >>> s = PhaseSpace('A-B-C', load=None)
            >>> s.phases = [ pa, pb, pc, p1, p2 ]
            >>> s.spaces
            [['C', 'B'], ['A', 'B']]

        """
        if self._spaces:
            return self._spaces
        spaces = set([ frozenset(p.space) for p in self.phase_dict.values() ])
        spaces = [ space for space in spaces if not 
                any([ space < space2 for space2 in spaces ])]
        self._spaces = list(map(list, spaces))
        return self._spaces

    def find_stable(self):
        stable = set()
        for space in self.spaces:
            subspace = self.get_subspace(space)
            stable |= set(subspace.stable)
        self._stable = stable
        return stable

    _dual_spaces = None
    @property
    def dual_spaces(self):
        """
        List of sets of elements, such that any possible tie-line
        between two phases in phases is contained in at least one
        set, and no set is a subset of any other.
        """
        if self._dual_spaces is None:
            self._dual_spaces = self.heap_structure_spaces()
        return self._dual_spaces

    def heap_structure_spaces(self):
        if len(self.spaces) == 1:
            return self.spaces
        heap = Heap()
        for i, (c1, c2) in enumerate(itertools.combinations(self.spaces, r=2)):
            heap.add(set(c1 + c2))
        return heap.sequences

    def get_dual_spaces(self):
        if len(self.spaces) == 1:
            return self.spaces

        dual_spaces = []
        imax = len(self.spaces)**2 / 2
        spaces = sorted(self.spaces, key=lambda x: -len(x))
        for i, (c1, c2) in enumerate(itertools.combinations(spaces, r=2)):
            c3 = frozenset(c1 + c2)
            if c3 in sizes[n]:
                break
            for j, c4 in enumerate(dual_spaces):
                if c3 <= c4:
                    break
                elif c4 < c3:
                    dual_spaces[j] = c3
                    break
            else:
                dual_spaces.append(c3)

        self._dual_spaces = dual_spaces
        return self._dual_spaces

    def find_tie_lines(self):
        phases = self.phase_dict.values()
        indict = dict((k, v) for v, k in enumerate(phases))
        adjacency = np.zeros((len(indict), len(indict)))
        for space in self.dual_spaces:
            subspace = self.get_subspace(space)
            for p1, p2 in subspace.tie_lines:
                i1, i2 = sorted([indict[p1], indict[p2]])
                adjacency[i1, i2] = 1
        tl = set( (phases[i], phases[j]) for i, j in 
                zip(*np.nonzero(adjacency)) )
        self._tie_lines = tl
        return tl

    @property
    def stable(self):
        """
        List of stable phases
        """
        if self._stable is None:
            self.hull
        return self._stable

    @property
    def unstable(self):
        """
        List of unstable phases.
        """
        if self._stable is None:
            self.hull
        return [ p for p in self.phases if
            ( not p in self.stable ) and self.in_space(p) ]

    _tie_lines = None
    @property
    def tie_lines(self):
        """
        List of length 2 tuples of phases with tie lines between them
        """
        if self._tie_lines is None:
            self.hull
        return [ list(tl) for tl in self._tie_lines ]

    @property
    def tie_lines_list(self):
        return list(self.tie_lines)

    @property
    def hull(self):
        """
        List of facets of the convex hull.
        """
        if self._hull is None:
            self.get_hull()
        return list(self._hull)

    def get_hull(self):
        if any( len(b) > 1 for b in self.bounds ):
            points = self.get_hull_points()
            self.get_qhull(phases=points)
        else:
            self.get_qhull()

    @property
    def hull_list(self):
        return list(self.hull)

    _graph = None
    @property
    def graph(self):
        """
        :mod:`networkx.Graph` representation of the phase space.
        """
        if self._graph:
            return self._graph
        graph = nx.Graph()
        graph.add_edges_from(self.tie_lines)
        self._graph = graph
        return self._graph

    _cliques = None
    @property
    def cliques(self):
        """
        Iterator over maximal cliques in the phase space. To get a list of
        cliques, use list(PhaseSpace.cliques).
        """
        if self._cliques is None:
            self.find_cliques()
        return self._cliques

    def find_cliques(self):
        self._cliques = nx.find_cliques(self.graph)
        return self._cliques

    def in_space(self, composition):
        """
        Returns True, if the composition is in the right elemental-space 
        for this PhaseSpace.

        Examples::

            >>> space = PhaseSpace('Fe-Li-O')
            >>> space.in_space('LiNiO2')
            False
            >>> space.in_space('Fe2O3')
            True

        """

        if self.bounds is None:
            return True
        if isinstance(composition, Phase):
            composition = composition.comp
        elif isinstance(composition, str):
            composition = parse_comp(composition)

        if set(composition.keys()) <= self.space:
            return True
        else:
            return False

    def in_bounds(self, composition):
        """
        Returns True, if the composition is within the bounds of the phase space

        Examples::

            >>> space = PhaseSpace('Fe2O3-NiO2-Li2O')
            >>> space.in_bounds('Fe3O4')
            False
            >>> space.in_bounds('Li5FeO8')
            True

        """
        if self.bounds is None:
            return True
        if isinstance(composition, Phase):
            composition = composition.unit_comp
        elif isinstance(composition, str):
            composition = parse_comp(composition)

        if not self.in_space(composition):
            return False

        composition = dict( (k,v) for k,v in composition.items() if k in
                self.bound_elements )
        composition = unit_comp(composition)

        try:
            c = self.coord(composition)
            if len(self.bounds) < len(self.space):
                comp = self.comp(c)
                if set(comp.keys()) != set(composition.keys())-set(self.mus.keys()):
                    return False
                if not all([abs(comp.get(k,0)- composition.get(k,0)) < 1e-3 for k in
                                                   self.bound_elements]):
                    return False
        except PhaseSpaceError:
            return False
        return True

    ### analysis stuff

    def get_qhull(self, phases=None, mus={}):
        """
        Get the convex hull for a given space.
        """
        if phases is None: ## ensure there are phases to get the hull of
            phases = self.phase_dict.values()

        ## ensure that all phases have negative formation energies
        _phases = []
        for p in phases:
            if not p.use:
                continue
            if self.phase_energy(p) > 0:
                continue
            if not self.in_bounds(p):
                continue
            _phases.append(p)

        phases = _phases

        phase_space = set()
        for p in phases:
            phase_space |= p.space

        A = []
        for p in phases:
            A.append(list(self.coord(p))[1:] + [self.phase_energy(p)])

        dim = len(A[0])
        for i in range(dim):
            tmparr = [ 0 if a != i-1 else 1 for a in range(dim) ]
            if not tmparr in A:
                A.append(tmparr)

        A = np.array(A)
        if len(A) == len(A[0]):
            self._hull = set([frozenset([ p for p in phases])])
            self._tie_lines = set([ frozenset([k1, k2]) for k1, k2 in
                    itertools.combinations(phases, r=2) ])
            self._stable = set([ p for p in phases])
            return
        conv_hull = ConvexHull(A)

        hull = set()
        tie_lines = set()
        stable = set()
        for facet in conv_hull.simplices:
            ### various exclusion rules
            if any([ ind >= len(phases) for ind in facet ]):
                continue

            if all( phases[ind].energy == 0 for ind in facet
                    if ind < len(phases)):
                continue

            dim = len(facet)
            face_matrix = np.array([ A[i] for i in facet ])
            face_matrix[:, -1] = 1
            v = np.linalg.det(face_matrix)

            if abs(v) < 1e-8:
                continue

            face = frozenset([ phases[ind] for ind in facet
                if ind < len(phases)])

            stable |= set(face)
            tie_lines |= set([ frozenset([k1, k2]) for k1, k2 in
                    itertools.combinations(face, r=2)])
            hull.add(Equilibrium(face))

        self._hull = hull
        self._tie_lines = tie_lines
        self._stable = stable
        return hull

    renderer = None
    @property
    def phase_diagram(self, **kwargs):
        """Renderer of a phase diagram of the PhaseSpace"""
        if self.renderer is None:
            self.get_phase_diagram(**kwargs)
        return self.renderer

    @property
    def neighboring_equilibria(self):
        neighbors = []
        for eq1, eq2 in itertools.combinations(self.hull, r=2):
            if eq1.adjacency(eq2) == 1:
                neighbors.append([eq1, eq2])
        return neighbors

    def get_phase_diagram(self, **kwargs):
        """
        Creates a Renderer attribute with appropriate phase diagram components.

        Examples::

            >>> space = PhaseSpace('Fe-Li-O')
            >>> space.get_renderer()
            >>> plt.show()

        """
        self.renderer = Renderer()
        if self.shape == (1,0):
            self.make_as_binary(**kwargs)
        elif self.shape == (2,0):
            self.make_as_ternary(**kwargs)
        elif self.shape == (3,0):
            self.make_as_quaternary(**kwargs)
        elif self.shape[0] > 3:
            ps = PhaseSpace('-'.join(self.space), data=self.data,
                    load=None)
            ps.renderer = Renderer()
            ps.make_as_graph(**kwargs)
            self.renderer = ps.renderer
        else:
            raise NotImplementedError
            
    def make_as_binary(self, **kwargs):
        """
        Construct a binary phase diagram (convex hull) and write it to a
        :mod:`~qmpy.Renderer`.

        Examples::
            
            >>> s = PhaseSpace('Fe-P')
            >>> r = s.make_as_binary()
            >>> r.plot_in_matplotlib()
            >>> plt.show()

        """

        xlabel = '%s<sub>x</sub>%s<sub>1-x</sub>' % (
                format_comp(self.bounds[0]),
                format_comp(self.bounds[1]))
        xaxis = Axis('x', label=xlabel)
        xaxis.min, xaxis.max = (0, 1)
        yaxis = Axis('y', label='Delta H', units='eV/atom')
        self.renderer.xaxis = xaxis
        self.renderer.yaxis = yaxis

        for p1, p2 in self.tie_lines:
            pt1 = Point([self.coord(p1)[0], self.phase_energy(p1)])
            pt2 = Point([self.coord(p2)[0], self.phase_energy(p2)])
            self.renderer.lines.append(Line([pt1, pt2], color='grey'))

        points = []
        for p in self.unstable:
            if not p.use:
                continue
            if self.phase_energy(p) > 0:
                continue
            if not self.in_bounds(p):
                continue
            x = self.coord(p.unit_comp)[0]
            pt = Point([x, self.phase_energy(p)], label=p.label)
            points.append(pt)

        self.renderer.point_collections.append(PointCollection(points,
            fill=1,
            color='red'))

        points = []
        for p in self.stable:
            if not self.in_bounds(p):
                continue
            x = self.coord(p.unit_comp)[0]
            pt = Point([x, self.phase_energy(p)], label=p.label)
            if p.show_label:
                self.renderer.text.append(Text(pt, p.name))
            points.append(pt)
        self.renderer.point_collections.append(PointCollection(points,
                                               fill=True, color='green'))

        self.renderer.options['grid']['hoverable'] = True
        self.renderer.options['tooltip'] = True
        self.renderer.options['tooltipOpts'] = {'content': '%label'}

    def make_as_ternary(self, **kwargs):
        """
        Construct a ternary phase diagram and write it to a
        :mod:`~qmpy.Renderer`.

        Examples::
            
            >>> s = PhaseSpace('Fe-Li-O-P')
            >>> r = s.make_as_quaternary()
            >>> r.plot_in_matplotlib()
            >>> plt.show()

        """

        for p1, p2 in self.tie_lines:
            pt1 = Point(coord_to_gtri(self.coord(p1)))
            pt2 = Point(coord_to_gtri(self.coord(p2)))
            line = Line([pt1, pt2], color='grey')
            self.renderer.lines.append(line)

        points = []
        for p in self.unstable:
            if not self.in_bounds(p):
                continue
            if self.phase_dict[p.name] in self.stable:
                continue
            ##pt = Point(coord_to_gtri(self.coord(p)), label=p.label)
            options = {'hull_distance': p.stability}
            pt = Point(coord_to_gtri(self.coord(p)), label=p.label, **options)
            points.append(pt)
        self.renderer.point_collections.append(PointCollection(points,
            fill=True, color='red'))

        self.renderer.options['xaxis']['show'] = False
        points = []
        for p in self.stable:
            if not self.in_bounds(p):
                continue
            pt = Point(coord_to_gtri(self.coord(p)), label=p.label)
            if p.show_label:
                self.renderer.add(Text(pt, p.name))
            points.append(pt)
        self.renderer.point_collections.append(PointCollection(points,
            fill=True,
            color='green'))

        self.renderer.options['grid']['hoverable'] = True, 
        self.renderer.options['grid']['borderWidth'] = 0
        self.renderer.options['grid']['margin'] = 4
        self.renderer.options['grid']['show'] = False
        self.renderer.options['tooltip'] = True

    def make_as_quaternary(self, **kwargs):
        """
        Construct a quaternary phase diagram and write it to a
        :mod:`~qmpy.Renderer`.

        Examples::
            
            >>> s = PhaseSpace('Fe-Li-O-P')
            >>> r = s.make_as_quaternary()
            >>> r.plot_in_matplotlib()
            >>> plt.show()

        """
        #plot lines
        for p1, p2 in self.tie_lines:
            pt1 = Point(coord_to_gtet(self.coord(p1)))
            pt2 = Point(coord_to_gtet(self.coord(p2)))
            line = Line([pt1, pt2], color='grey')
            self.renderer.add(line)

        #plot compounds
        ### < Mohan
        # Use phase_dict to collect unstable phases, which will 
        # return one phase per composition
        points = []
        for c, p in self.phase_dict.items():
            if not self.in_bounds(p):
                continue
            if p in self.stable:
                continue
            label = '{}<br> hull distance: {:.3f} eV/atom<br> formation energy: {:.3f} eV/atom'.format(
                p.name, p.stability, p.energy
            )
            pt = Point(coord_to_gtet(self.coord(p)), label=label)
            points.append(pt)
        self.renderer.add(PointCollection(points, 
                                          color='red', label='Unstable'))

        points = []
        for p in self.stable:
            if not self.in_bounds(p):
                continue
            label = '%s:<br>- ' % p.name
            label += ' <br>- '.join(o.name for o in self.graph[p].keys())
            pt = Point(coord_to_gtet(self.coord(p)), label=label)
            points.append(pt)
            if p.show_label:
                self.renderer.add(Text(pt, format_html(p.comp)))
        self.renderer.add(PointCollection(points, 
                                          color='green', label='Stable'))

        self.renderer.options['grid']['hoverable'] = True, 
        self.renderer.options['grid']['borderWidth'] = 0
        self.renderer.options['grid']['show'] = False
        self.renderer.options['tooltip'] = True

    def make_as_graph(self, **kwargs):
        """
        Construct a graph-style visualization of the phase diagram.
        """
        G = self.graph
        positions = nx.drawing.nx_agraph.pygraphviz_layout(G)
        for p1, p2 in self.tie_lines:
            pt1 = Point(positions[p1])
            pt2 = Point(positions[p2])
            line = Line([pt1, pt2], color='grey')
            self.renderer.add(line)

        points = []
        for p in self.stable:
            label = '%s:<br>' % p.name
            for other in G[p].keys():
                label += '  -%s<br>' % other.name
            pt = Point(positions[p], label=label)
            points.append(pt)
            if p.show_label:
                self.renderer.add(Text(pt, p.name))
        pc = PointCollection(points, color='green')
        self.renderer.add(pc)

        self.renderer.options['grid']['hoverable'] = True
        self.renderer.options['grid']['borderWidth'] = 0
        self.renderer.options['grid']['show'] = False
        self.renderer.options['tooltip'] = True
