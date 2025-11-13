from typing import List
from neo4j import GraphDatabase
from core.config import settings

class Neo4jRepository:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def _execute_write(self, query, **params):
        # Using a managed transaction for robustness
        with self._driver.session() as session:
            return session.execute_write(lambda tx: tx.run(query, **params).data())

    def add_entities_and_relationships(self, extracted_data: dict, source_episode_id: str):
        """
        Adds entities and relationships from a structured extraction to the graph.
        It also links them to the source episodic memory.
        """
        entities = extracted_data.get("entities", [])
        relationships = extracted_data.get("relationships", [])

        if not entities:
            print("No entities found in extraction data. Skipping graph population.")
            return

        # Create an Episode node to represent the source of this information
        self._execute_write(
            "MERGE (e:Episode {id: $episode_id})",
            episode_id=source_episode_id
        )

        # Use MERGE to create or update entity nodes and link them to the episode
        for entity in entities:
            # Ensure entity name and type are not None
            if entity.get('name') and entity.get('type'):
                query = """
                MERGE (n:Entity {name: $name})
                ON CREATE SET n.type = $type
                ON MATCH SET n.type = $type
                WITH n
                MATCH (e:Episode {id: $episode_id})
                MERGE (n)-[:MENTIONED_IN]->(e)
                """
                self._execute_write(query, name=entity['name'], type=entity['type'], episode_id=source_episode_id)

        # Create relationships between the entities
        for rel in relationships:
            # Ensure all parts of the relationship are present
            if rel.get('source') and rel.get('target') and rel.get('type'):
                # Sanitize relationship type for Cypher: must be alphanumeric + underscore
                rel_type = ''.join(c for c in rel['type'].upper() if c.isalnum() or c == '_')
                query = f"""
                MATCH (source:Entity {{name: $source_name}})
                MATCH (target:Entity {{name: $target_name}})
                MERGE (source)-[r:{rel_type}]->(target)
                """
                self._execute_write(query, source_name=rel['source'], target_name=rel['target'])
        
        print(f"Added {len(entities)} entities and {len(relationships)} relationships to the knowledge graph.")

    def _execute_read(self, query, **params):
        with self._driver.session() as session:
            return session.execute_read(lambda tx: tx.run(query, **params).data())

    def search_semantic_entities(self, entity_query: str, top_k: int = 5) -> List[dict]:
        """
        Finds an entity in the graph and returns all its direct relationships
        and the entities they connect to, formatted as factual statements.
        """
        print(f"Searching Neo4j for facts related to: '{entity_query}'")
        
        # This Cypher query finds the starting entity (e.g., 'Project Aquila'),
        # then finds all nodes connected to it by any relationship,
        # and returns the triplet: source_node, relationship_type, target_node.
        query = """
        MATCH (source:Entity {name: $entity_name})-[r]-(target:Entity)
        RETURN source.name AS source, type(r) AS relationship, target.name AS target
        LIMIT $limit
        """
        
        try:
            results = self._execute_read(query, entity_name=entity_query, limit=top_k)
            
            if not results:
                print(f"No semantic relationships found for entity '{entity_query}'.")
                return []

            # Format the graph triplets into human-readable facts
            facts = []
            for record in results:
                # Make the fact more natural, e.g., "Project Aquila HAS_DEADLINE December 1st, 2025"
                fact_content = f"{record['source']} {record['relationship'].replace('_', ' ').lower()} {record['target']}."
                facts.append({
                    "content": f"Fact: {fact_content}",
                    "score": 0.99, # High score for a direct graph hit
                    "metadata": {
                        "source_node": record['source'],
                        "relationship": record['relationship'],
                        "target_node": record['target']
                    }
                })
            
            print(f"Found {len(facts)} semantic facts for '{entity_query}'.")
            return facts

        except Exception as e:
            print(f"Error searching Neo4j: {e}")
            return []

# Global instance
neo4j_repo = Neo4jRepository(
    uri=settings.NEO4J_URI,
    user=settings.NEO4J_USER,
    password=settings.NEO4J_PASSWORD
)