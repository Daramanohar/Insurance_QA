import config

current = config.PINECONE_NAMESPACE
if current:
    print(f'Current namespace: "{current}"')
else:
    print('Current namespace: (empty - using default namespace)')

