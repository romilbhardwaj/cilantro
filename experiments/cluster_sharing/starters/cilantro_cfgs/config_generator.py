# Reads a fixed config yaml and generates configurations specified in the POLICIES variable

POLICIES = ['mmf'] #, 'mmflearn', 'utilwelforacle', 'utilwelflearn', 'evoutil', 'egalwelforacle', 'egalwelflearn', 'evoegal', 'greedyegal', 'minerva', 'ernest', 'quasar', 'parties', 'multincadddec'] # Exclude propfair

with open('config_cilantro_scheduler_propfair.yaml') as f:
    # Read as str
    config = f.read()


# Replace instances of 'propfair' with the policy name
for policy in POLICIES:
    new_config = config.replace('propfair', policy)
    with open('config_cilantro_scheduler_{}.yaml'.format(policy), 'w') as f:
        f.write(new_config)
