'''
Created on 19/07/2016

@author: achim
'''

from collections import namedtuple
from ruleProgramming import impact_factory

impact = namedtuple("impact", ["time", "timeseries", "operation"])

# the idea of an impact is to influence how the next point in time in a time
# series is determined. The impacts are "lambda" functions of a single argument
# (the value before, returning the value after). The list of impacts determines
# the order of lambda functions applied for one timeseries at one point in time
# (i.e. one "impact")


# this impact operation is easier to follow/debug
class impact_op:
    def __init__(self, op, value):
        self.op = op
        self.value = value

    def __call__(self, x):
        if self.op == "add":
            return x+self.value
        elif self.op == "mul":
            return x*self.value
        elif self.op == "set":
            return self.value+0
        raise ValueError("unknown operation")

    def __repr__(self):
        # make a readable representation for debugging
        return "<%s op='%s' value='%s'>" % (type(self).__name__,
                                            self.op, self.value)


def impact_factory(t, name, operation, value):
    # this one will have a better string representation
    return impact(t, name, impact_op(operation, value))


def impact_factory1(t, name, operation, value):
    # this implementation is faster!

    from functools import partial
    from operator import add, mul

    if operation == "add":
        op = partial(add, value)
    elif operation == "mul":
        op = partial(mul, value)
    elif operation == "set":
        def op(x):
            return value
    else:
        raise ValueError("unknown operation")

    return impact(t, name, op)


class timeseries:
    # a time series object holding many time series in a dictionary
    # apply impacts step by step

    # this class needs a way of freezing the time series till a certain point
    # and then continuing

    # define meaning of values more rigorously (to avoid one off problems):

    # each value is the resulting figure at the end of the month
    # (i.e. all impacts of this month applied)
    # the first value is the initial value (i.e. the end of the time interval
    # before the first impacts are applied).

    def __init__(self, initial, default):
        self.data = {k: [v] for k, v in initial.items()}
        # this is the default impact
        # an empty list will result in repeating the value
        # (side effect of functools.reduce)
        self.default = {k: [] for k in initial.keys()}
        self.default.update({k: [i] for k, i in default.items()})

    def apply_next(self, impacts):
        # step one forward in the time series

        t = next(len(v)-1 for v in self.data.values())

        from functools import reduce

        # start off with the default impacts
        allImpacts = {k: v.copy() for k, v in self.default.items()}

        # copy the new impacts
        for ti, ts, op in impacts:
            if ti == t:
                allImpacts[ts].append(op)

        # and apply them
        for ts, ops in allImpacts.items():
            s = self.data[ts]
            s.append(reduce(lambda x, o: o(x),
                            ops,
                            s[-1]))

    def copy(self):
        # copy the time series object, but keep the default impacts the same
        ts = timeseries({}, {})
        ts.data = {k: v.copy() for k, v in self.data.items()}
        ts.default = self.default
        return ts


def simulate(initial, projects):
    simulation_duration = 120

    # this needs sorting with project/phase priority
    planned_phases = list(sorted((ph for p in projects for ph in p.phases),
                                 key=lambda x: x.priority))

    t0 = {}
    actual_impacts = []

    # take impacts and collect timelines:
    timeline_names = set(imp.timeseries
                         for p in projects
                         for ph in p.phases
                         for opt in ph.actualImpacts
                         for imp in opt)
    for tn in timeline_names:
        t0[tn] = 0.0

    # create a status timeline for each phase
    for p in projects:
        for ph in p.phases:
            t0["status_%d" % ph.id] = -1

    # create timeline for each deliverable
    # create timeline for each resource

    # create timeline for funds and add refill impacts
    t0["investment fund"] = 0
    for i in range(simulation_duration):
        if i % 12 == 0:
            actual_impacts.append(impact_factory(i, "investment fund",
                                                 "set", 50e6))

    t0.update(initial)

    for i in range(simulation_duration):
        # nothing to schedule anymore? done!
        if not planned_phases:
            break

        # dispatch phases in time step i

        # order by priority
        dispatched_phases = []
        dispatched_aImpacts = []
        dispatched_pImpacts = []

        # clean up which timeline needs to be re-calculated to which length!
        # only keep actual impacts for future, as the others are no longer used
        base_timeseries = timeseries(initial=t0, default={})
        for _ in range(i):
            base_timeseries.apply_next(actual_impacts)

        new_timeseries = base_timeseries.copy()
        while (len(new_timeseries.data["investment fund"]) <=
               simulation_duration+1):
            new_timeseries.apply_next(dispatched_pImpacts)

        # in the end, this is a packing problem!
        # https://en.wikipedia.org/wiki/Knapsack_problem
        for p in planned_phases:
            # if a project/phase fulfilled: earliest start and dependencies
            # decide whether dependencies are fulfilled and whether schedule it

            if i >= p.start and all(new_timeseries.data["status_%d" %
                                                        pid][i+1] >= 100
                                    for pid in p.dependencies):

                # this loop needs to change:
                # and deal with dependencies, which are fulfilled during the
                # loop execution

                # try out different options!
                for pImpacts, aImpacts in zip(p.plannedImpacts,
                                              p.actualImpacts):

                    # try out different impacts, but stay with the planned ones
                    test_timeseries = base_timeseries.copy()
                    # move the planned impacts to the right time
                    test_impacts = (actual_impacts +
                                    dispatched_pImpacts +
                                    [imp._replace(time=imp.time+i)
                                     for imp in pImpacts])

                    # try out whether this phase fits in!
                    while (len(test_timeseries.data["investment fund"]) <=
                           simulation_duration+1):
                        test_timeseries.apply_next(test_impacts)

                    # see whether all business rules are ok
                    # see whether resources are ok as well
                    # WARNING: an accidentally negative resource might block
                    # dispatching new "unrelated" phases: don't change resource
                    # amount with "eventuality".

                    if (all(v >= 0
                            for v in test_timeseries.data["investment fund"]
                            ) and
                        all(all(v >= 0 for v in ts)
                            for n, ts in test_timeseries.data.items()
                            if n.startswith("resource_"))):

                        dispatched_phases.append(p)
                        # lock them in
                        dispatched_pImpacts.extend(
                                           imp._replace(time=imp.time+i)
                                           for imp in pImpacts)
                        dispatched_aImpacts.extend(
                                           imp._replace(time=imp.time+i)
                                           for imp in aImpacts)
                        new_timeseries = test_timeseries
                        # and take the first one
                        break

        # and move the phase out of the planned_phases
        # and accept the phases actual impacts
        # that means the full (future) actual impact is already visible.
        for p in dispatched_phases:
            planned_phases.remove(p)
            actual_impacts.extend(dispatched_aImpacts)

    # start over fresh! (could be based on the base time series...)
    while len(base_timeseries.data["investment fund"]) <= 120:
        base_timeseries.apply_next(actual_impacts)

    return base_timeseries

project = namedtuple("project", ["name", "phases"])

# planned impacts have a list of impact lists (called options)
# so do actualImpacts (same list length)
projectphase = namedtuple("projectphase", ["id", "name", "dependencies",
                                           "priority", "start",
                                           "plannedImpacts", "actualImpacts"])

# need something, which counts up!
# to get the projectphase.id populated
phase_id_counter = 0


# build up functions, which create projects and their phases,
# manage dependencies

# the first one just creates the project container
def create_project(name):
    return project(name=name, phases=[])


# a project has different phases, the following phase depends on the preceding
# if a project depends on another one, it depends on all phases being finished
def add_phase(proj, start, options):

    # start: earliest start time for this phase
    # other project dependencies

    # a phase has different options
    # append the phase
    # each option has:
    # investment, impact on service, resources to draw from,
    # phase length

    # a phase sets the status at start and end
    # allocates resources at start
    # frees resources at end
    global phase_id_counter

    phase_id_counter = phase_id_counter + 1
    ph_id = phase_id_counter + 0  # copy!

    # let's define phase options. These are the properties
    # which determine the impacts
    ph = projectphase(id=ph_id,
                      name=proj.name+"_%d" % len(proj.phases),
                      dependencies=([proj.phases[-1].id]
                                    if proj.phases else []),
                      priority=0,  # todo
                      start=start,
                      plannedImpacts=[],
                      actualImpacts=[])

    for opt in options:
        add_option(ph, opt)

    # delivers the impact at end
    # decreases the investment budget
    proj.phases.append(ph)


# the option is a dictionary with:
# length: the project length in months (mandatory)
# investment: how much comes from investment fund (equally over months)
# resources: dictionary with resource names and values to be acquired/released
# impact: list of impacts (the time 0..length can schedule the impact at any
#         time during the project)

def add_option(phase, option):
    impF = impact_factory
    ph_id = phase.id
    optCounter = len(phase.plannedImpacts)

    # todo: this comes out of option
    phaseLen = option.get("length", 3)
    investmentCost = option.get("investment", 0)
    resources = option.get("resources", {})  # resource name, number
    output_impact = option.get("impact", [])

    # this sets the status, acquires resources
    # (by withdrawing them from the public pool)
    impStart = ([impF(t=0, name="status_%d" % ph_id,
                      operation="set", value=optCounter)] +
                [impF(t=0, name="resource_%s" % r,
                      operation="add", value=-v)
                 for r, v in resources.items()])

    impMiddle = [impF(t=i, name="investment fund",
                      operation="add", value=-investmentCost/phaseLen)
                 for i in range(phaseLen)]

    impEnd = (output_impact +
              [impF(t=phaseLen, name="status_%d" % ph_id,
                    operation="set", value=optCounter+100)] +
              [impF(t=phaseLen, name="resource_%s" % r,
                    operation="add", value=v)
               for r, v in resources.items()])

    phase.plannedImpacts.append(impStart + impMiddle + impEnd)
    phase.actualImpacts.append(impStart + impMiddle + impEnd)


def create_project_dependency(p1, p2):
    # p2 will depend on completion of the last phase of p1
    # both projects will need some phases
    # to be tested!
    p2.phases[0].dependencies.append(p1.phases[-1].id)

# eventuality:
# schedule a modification of the planned project
# and specify their likelihood
# i.e. this function modifies the actualImpact data-set
# this could be: an outcome could be reduced
# the project takes longer

# eventuality
# likelihood, modification, phase_id (i.e. project, phase)
# the eventuality will be applied to any of the options.


def get_phase_id(projects, project_name, phase_no):
    # phase no starts with 0

    return next(p.phases[phase_no].id for p in projects
                if p.name == project_name)


def no_modification(plannedImpacts):
    return plannedImpacts


def delay_by_one(plannedImpacts):
    # which adds one "do nothing" month in the front
    delay_time = 1

    status_name, status_value = next((imp.timeseries, imp.operation(0))
                                     for imp in plannedImpacts
                                     if (imp.timeseries.startswith(
                                            "status_") and imp.time == 0))

    return ([impact_factory(0, status_name, "set", status_value)]*delay_time +
            [imp._replace(time=imp.time+delay_time) for imp in plannedImpacts])


def phase_more_costly(plannedImpacts):

    cost_increase = 1.1

    return [impact_factory(imp.time, imp.timeseries,
                           "add", imp.operation(0.0)*cost_increase)
            if imp.timeseries == "investment fund" else imp
            for imp in plannedImpacts]


def test_modifications():

    p = create_project("foo")
    add_phase(p, 0, [{"length": 2,
                      "investment": 1e6
                      }])

    impacts = p.phases[0].plannedImpacts[0]
    assert len(impacts) == 4  # 3 for status and 2 for investment?!
    assert sum(-imp.operation(0) for imp in impacts
               if imp.timeseries == "investment fund") == 1e6

    assert impacts == no_modification(impacts)
    assert len(phase_more_costly(impacts)) == len(impacts)
    assert len(delay_by_one(impacts)) == len(impacts)+1
    assert sum(-imp.operation(0) for imp in phase_more_costly(impacts)
               if imp.timeseries == "investment fund") == 1.1e6

    from functools import reduce

    print(reduce(lambda i, m: m(i),
                 [delay_by_one, no_modification, phase_more_costly],
                 impacts))


def test_projects():
    # set up a project
    p = create_project("foo")

    opt1_ph1 = {"length": 2, "resources": {"analyst": 1},
                "impact": [impact_factory(2, "output", "add", 1)]}
    opt2_ph1 = {"length": 3, "investment": 1e6}

    add_phase(p, start=3, options=[opt1_ph1, opt2_ph1])

    opt1_ph2 = {"length": 2, "investment": 1e6}
    opt2_ph2 = {"length": 3}
    add_phase(p, start=10, options=[opt1_ph2, opt2_ph2])

    # set the resources and outputs
    initial = {"resource_analyst": 0,
               "output": 1}

    ts = simulate(initial, [p])
    for n, s in ts.data.items():
        if n.startswith("status") or n.startswith("resource_"):
            print(n, s)


def test_eventualities():
    from itertools import product
    from operator import mul
    from functools import reduce
    from collections import defaultdict

    l1 = 0.1
    l2 = 0.2
    p = create_project("foo")
    add_phase(p, 0, [{"length": 2,
                      "investment": 1e6
                      }])

    projects = [p]

    p0 = get_phase_id(projects, "foo", 0)

    for scenario in product([(1.0-l1, p0, None), (l1, p0, delay_by_one)],
                            [(1.0-l2, p0, None), (l2, p0, phase_more_costly)]):

        likelihood = reduce(mul, (ll for ll, _, _ in scenario), 1.0)

        # collect modifications
        mods = defaultdict(list)

        # modify the planned impacts
        for _, phaseId, mod in scenario:
            if mod is not None:
                mods[phaseId].append(mod)

        for p in projects:
            for ph in p.phases:
                # do an in-place modification of the actual impacts list
                ph.actualImpacts.clear()
                if ph.id in mods:
                    # take the planned impacts and store the modified ones
                    ph.actualImpacts.extend(reduce(lambda i, m: m(i),
                                                   mods[ph.id],
                                                   opt)
                                            for opt in ph.plannedImpacts)
                else:
                    # copy
                    ph.actualImpacts.extend(ph.plannedImpacts)

        tss = simulate({}, projects)

        print(likelihood)
        print(tss.data["status_%d" % p0])


def test_simulate():

    impf = impact_factory

    imp1 = [impf(0, "status_1", "set", 1),
            impf(3, "status_1", "set", 100),
            impf(0, "investment fund", "add", -1e6)]

    ph1 = projectphase(id=1, name="", priority=1,
                       dependencies=[], start=3,
                       plannedImpacts=[imp1], actualImpacts=[imp1])

    imp2 = [impf(0, "status_2", "set", 1),
            impf(5, "status_2", "set", 100),
            impf(0, "investment fund", "add", -2e6)]
    ph2 = projectphase(id=2, name="", priority=1,
                       dependencies=[1], start=1,
                       plannedImpacts=[imp2], actualImpacts=[imp2])

    p = project(name="", phases=[ph1, ph2])

    ts = simulate({}, [p])
    assert ts.data["status_1"][-1] == 100
    assert ts.data["status_2"][-1] == 100
#     print(ts.data["investment fund"])
#     print(ts.data["status_1"])
#     print(ts.data["status_2"])


def test_add():
    tss = timeseries({"x": 0}, {})

    impF = impact_factory

    impacts = [impF(0, "x", "add", -1.0),
               impF(5, "x", "add", 1.0)]

    for _ in range(10):
        tss.apply_next(impacts)
    assert tss.data["x"][-1] == 0


def test_ts_default():
    tss = timeseries({"t": 0},
                     {"t": impact_factory("", 0, "add", 1).operation})
    for _ in range(10):
        tss.apply_next([])

    assert "t" in tss.data
    assert tss.data["t"] == list(range(0, 11))
