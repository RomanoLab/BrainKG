import time
import xml.etree.ElementTree as ET
from collections import defaultdict
import torch
from torch_geometric.data import HeteroData
import re

# Namespace dictionary for XML parsing
NS = {
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
    'owl': 'http://www.w3.org/2002/07/owl#',
    'xsd': 'http://www.w3.org/2001/XMLSchema#',
}

SCHEMA_TAGS = {'Class', 'ObjectProperty', 'DatatypeProperty', 'AnnotationProperty', 'Ontology', 'Restriction', 'AllDisjointClasses'}
IGNORE_RELS = {'domain', 'range', 'subClassOf', 'subPropertyOf', 'inverseOf', 'imports', 'type'}
GENERIC_TYPES = {'NamedIndividual', 'Description', 'Thing'}

def sanitize_label(label: str) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', label)
    if sanitized and not sanitized[0].isalpha():
        sanitized = 'N_' + sanitized
    return sanitized or 'Node'

def extract_fragment(iri: str) -> str:
    iri = iri.strip()
    if '#' in iri: return iri.split('#')[-1]
    elif '/' in iri: return iri.split('/')[-1]
    elif ':' in iri and not iri.startswith('http'): 
        return iri.split(':')[-1]
    return iri

def normalize_iri(iri: str, base_namespace: str) -> str:
    iri = iri.strip()
    if iri.startswith('#'):
        iri = base_namespace.rstrip('#') + iri
    return iri

def build_hetero_graph(rdf_file: str) -> HeteroData:
    print(f"Parsing RDF file: {rdf_file} ...")
    start_time = time.time()
    
    tree = ET.parse(rdf_file)
    root = tree.getroot()
    
    ont_elem = root.find('owl:Ontology', NS)
    base_namespace = ont_elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about', '') if ont_elem is not None else ''
    if base_namespace and not base_namespace.endswith('#'):
        base_namespace += '#'
    
    # Use a master dictionary to accumulate ALL types across ALL XML blocks
    node_types_map = defaultdict(set)
    edges_raw = []

    print("Extracting nodes and relationships...")
    for elem in root:
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        
        if tag in SCHEMA_TAGS:
            continue
            
        iri = elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
        if not iri:
            iri = elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}ID')
            if iri: iri = '#' + iri
        if not iri: 
            continue
            
        iri = normalize_iri(iri, base_namespace)
        
        # If the XML tag itself is the Class (e.g. <Drug rdf:about="...">)
        if tag not in GENERIC_TYPES:
            node_types_map[iri].add(tag)
            
        for child in elem:
            child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            
            # Explicit <rdf:type rdf:resource=".../Drug" />
            if child_tag == 'type':
                type_iri = child.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
                if type_iri:
                    t_name = extract_fragment(type_iri)
                    if t_name not in SCHEMA_TAGS and t_name not in GENERIC_TYPES:
                        node_types_map[iri].add(t_name)
            
            # Catch relationships (Edges)
            elif '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource' in child.attrib:
                if child_tag in IGNORE_RELS:
                    continue
                    
                target_iri = child.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource', '')
                target_iri = normalize_iri(target_iri, base_namespace)
                if target_iri:
                    edges_raw.append((iri, sanitize_label(child_tag), target_iri))

    print("Consolidating node types...")
    # Add empty sets for target nodes that were only ever found in edges
    for src_iri, rel, dst_iri in edges_raw:
        if src_iri not in node_types_map:
            node_types_map[src_iri] = set()
        if dst_iri not in node_types_map:
            node_types_map[dst_iri] = set()

    # Finalize one primary type per node
    node_iri_to_type = {}
    for iri, types in node_types_map.items():
        valid_types = [t for t in types if t not in GENERIC_TYPES and t not in SCHEMA_TAGS]
        if valid_types:
            node_iri_to_type[iri] = sanitize_label(sorted(valid_types)[0])
        else:
            node_iri_to_type[iri] = "Entity"

    # Group by type for PyTorch Geometric
    nodes_by_type = defaultdict(list)
    for iri, n_type in node_iri_to_type.items():
        nodes_by_type[n_type].append(iri)

    print("\nBuilding PyG HeteroData object...")
    data = HeteroData()
    node_maps = {} 
    
    for node_type, iris in nodes_by_type.items():
        node_maps[node_type] = {iri: i for i, iri in enumerate(iris)}
        data[node_type].num_nodes = len(iris)
        print(f"  {node_type}: {len(iris):,} nodes")

    edges_by_triplet = defaultdict(lambda: ([], []))
    for src_iri, rel, dst_iri in edges_raw:
        src_type = node_iri_to_type[src_iri]
        dst_type = node_iri_to_type[dst_iri]
        
        src_idx = node_maps[src_type][src_iri]
        dst_idx = node_maps[dst_type][dst_iri]
        
        triplet = (src_type, rel, dst_type)
        edges_by_triplet[triplet][0].append(src_idx)
        edges_by_triplet[triplet][1].append(dst_idx)

    for triplet, (srcs, dsts) in edges_by_triplet.items():
        data[triplet].edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        print(f"  {triplet}: {len(srcs):,} edges")

    print(f"\nProcessing complete in {time.time() - start_time:.2f}s")
    
    payload = {
        "graph": data,
        "node_maps": node_maps
    }
    return payload

if __name__ == "__main__":
    rdf_file = "brainkg-populated.rdf"
    out_file = "NeuroKB_PYG.pt"
    
    payload = build_hetero_graph(rdf_file)
    torch.save(payload, out_file)
    print(f"\nSaved graph and node mappings to {out_file}")