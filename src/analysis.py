import os
import pm4py


def import_xes(file_path):
    event_log = pm4py.read_xes(file_path)
    start_activities = pm4py.get_start_activities(event_log)
    end_activities = pm4py.get_end_activities(event_log)
    print("Start activities: {}\nEnd activities: {}".format(start_activities, end_activities))
    return event_log


def discover_dfg(event_log):
    return pm4py.discovery.discover_dfg(event_log)


def discover_performance_dfg(event_log):
    return pm4py.discovery.discover_performance_dfg(event_log)


def discover_petri_net_alpha(event_log):
    return pm4py.discovery.discover_petri_net_alpha(event_log)


def discover_petri_net_inductive(event_log):
    return pm4py.discovery.discover_petri_net_inductive(event_log)


def discover_petri_net_heuristics(event_log):
    return pm4py.discovery.discover_petri_net_heuristics(event_log)


def discover_petri_net_ilp(event_log):
    return pm4py.discovery.discover_petri_net_ilp(event_log)


def discover_declare(event_log):
    return pm4py.discovery.discover_declare(event_log)


def view_petri_net(net, initial_marking, final_marking):
    return pm4py.vis.view_petri_net(net, initial_marking, final_marking)


def view_dfg(dfg, start_activities=None, end_activities=None):
    return pm4py.vis.view_dfg(dfg, start_activities, end_activities)


def view_performance_dfg(performance_dfg, start_activities=None, end_activities=None):
    return pm4py.vis.view_performance_dfg(performance_dfg, start_activities, end_activities)


def view_case_duration_graph(event_log):
    return pm4py.vis.view_case_duration_graph(event_log)


def view_events_per_time_graph(event_log):
    return pm4py.vis.view_events_per_time_graph(event_log)


def view_performance_spectrum(event_log):
    return pm4py.vis.view_performance_spectrum(event_log)


def view_events_distribution_graph(event_log):
    return pm4py.vis.view_events_distribution_graph(event_log)


def view_transition_system(event_log):
    return pm4py.vis.view_transition_system(event_log)


def view_alignments(alignments):
    return pm4py.vis.view_alignments(alignments)


def conformance_diagnostics_token_based_replay(event_log, net, initial_marking, final_marking):
    return pm4py.conformance.conformance_diagnostics_token_based_replay(
        event_log, net, initial_marking, final_marking
    )


def conformance_diagnostics_alignments(event_log, net, initial_marking, final_marking):
    return pm4py.conformance.conformance_diagnostics_alignments(
        event_log, net, initial_marking, final_marking
    )


def fitness_token_based_replay(event_log, net, initial_marking, final_marking):
    return pm4py.conformance.fitness_token_based_replay(
        event_log, net, initial_marking, final_marking
    )


def fitness_alignments(event_log, net, initial_marking, final_marking):
    return pm4py.conformance.fitness_alignments(
        event_log, net, initial_marking, final_marking
    )


def get_start_activities(event_log):
    return pm4py.stats.get_start_activities(event_log)


def get_end_activities(event_log):
    return pm4py.stats.get_end_activities(event_log)


def get_event_attributes(event_log):
    return pm4py.stats.get_event_attributes(event_log)


def get_trace_attributes(event_log):
    return pm4py.stats.get_trace_attributes(event_log)


def get_event_attribute_values(event_log, attribute_key):
    return pm4py.stats.get_event_attribute_values(event_log, attribute_key)


def get_trace_attribute_values(event_log, attribute_key):
    return pm4py.stats.get_trace_attribute_values(event_log, attribute_key)


def get_variants(event_log):
    return pm4py.stats.get_variants(event_log)


def split_by_process_variant(event_log):
    return pm4py.stats.split_by_process_variant(event_log)


def get_variants_paths_duration(event_log):
    return pm4py.stats.get_variants_paths_duration(event_log)


def get_frequent_trace_segments(event_log):
    return pm4py.stats.get_frequent_trace_segments(event_log)


def get_case_arrival_average(event_log):
    return pm4py.stats.get_case_arrival_average(event_log)


def get_cycle_time(event_log):
    return pm4py.stats.get_cycle_time(event_log)


def get_all_case_durations(event_log):
    return pm4py.stats.get_all_case_durations(event_log)


def get_case_duration(event_log, case_id):
    return pm4py.stats.get_case_duration(event_log, case_id)


def get_stochastic_language(obj):
    return pm4py.stats.get_stochastic_language(obj)


def get_service_time(event_log):
    return pm4py.stats.get_service_time(event_log)


def sim_play_out(process_model, initial_marking, final_marking):
    return pm4py.sim.play_out(process_model, initial_marking, final_marking)


def sim_generate_process_tree(num_nodes):
    return pm4py.sim.generate_process_tree(num_nodes)


def abstract_dfg(event_log):
    return pm4py.llm.abstract_dfg(event_log)


def abstract_variants(event_log):
    return pm4py.llm.abstract_variants(event_log)


def abstract_log_attributes(event_log):
    return pm4py.llm.abstract_log_attributes(event_log)


def abstract_log_features(event_log):
    return pm4py.llm.abstract_log_features(event_log)


def abstract_case(case):
    return pm4py.llm.abstract_case(case)


def abstract_event_stream(event_log):
    return pm4py.llm.abstract_event_stream(event_log)


def abstract_temporal_profile(temporal_profile):
    return pm4py.llm.abstract_temporal_profile(temporal_profile)


def abstract_petri_net(net, initial_marking, final_marking):
    return pm4py.llm.abstract_petri_net(net, initial_marking, final_marking)


def abstract_declare(declare_model):
    return pm4py.llm.abstract_declare(declare_model)


def split_train_test(event_log, train_percentage=0.8):
    return pm4py.ml.split_train_test(event_log, train_percentage)






if __name__ == "__main__":
    dir_path = os.path.join("..", "logs")
    file_path = os.path.join(dir_path, "road_traffic.xes")
    log = import_xes(file_path)

    dfg, start_activities, end_activities = discover_dfg(log)
    performance_dfg, _, _ = discover_performance_dfg(log)
    net, initial_marking, final_marking = discover_petri_net_inductive(log)
    declare_model = discover_declare(log)
    token_replay = conformance_diagnostics_token_based_replay(
        log, net, initial_marking, final_marking
    )
    fitness_token = fitness_token_based_replay(log, net, initial_marking, final_marking)
    start_stats = get_start_activities(log)
    end_stats = get_end_activities(log)
    variants = get_variants(log)
    case_durations = get_all_case_durations(log)
    simulated_log = sim_play_out(net, initial_marking, final_marking)
    dfg_abstraction = abstract_dfg(log)
    variants_abstraction = abstract_variants(log)
    log_attributes_abstraction = abstract_log_attributes(log)
    petri_net_abstraction = abstract_petri_net(net, initial_marking, final_marking)
    declare_abstraction = abstract_declare(declare_model)

    print("DFG edges: {}".format(len(dfg)))
    print("Performance DFG edges: {}".format(len(performance_dfg)))
    print("Petri net places: {}".format(len(net.places)))
    print("DECLARE constraints: {}".format(len(declare_model)))
    print("Token-based replay cases: {}".format(len(token_replay)))
    print("Token-based fitness: {}".format(fitness_token))
    print("Start activities (stats): {}".format(start_stats))
    print("End activities (stats): {}".format(end_stats))
    print("Variants: {}".format(len(variants)))
    print("Case durations: {}".format(len(case_durations)))
    print("Simulated log traces: {}".format(len(simulated_log)))
    print("DFG abstraction length: {}".format(len(dfg_abstraction)))
    print("Variants abstraction length: {}".format(len(variants_abstraction)))
    print("Log attributes abstraction length: {}".format(len(log_attributes_abstraction)))
    print("Petri net abstraction length: {}".format(len(petri_net_abstraction)))
    print("DECLARE abstraction length: {}".format(declare_abstraction))

    if os.environ.get("PM4PY_VIEW", "") == "1":
        view_dfg(dfg, start_activities, end_activities)
        view_performance_dfg(performance_dfg, start_activities, end_activities)
        view_petri_net(net, initial_marking, final_marking)