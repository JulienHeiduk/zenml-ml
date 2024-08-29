from zenml.client import Client

artifact = Client().get_artifact_version('4f12d004-1a1f-453f-9321-a4da200345d4')
loaded_artifact = artifact.load()

print(loaded_artifact)