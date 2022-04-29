# The Multi-Party Dialogue Dataset (MPDD)

MPDD consists of two files, dialogue.json and metadata.json.

## dialogue.json

The file dialogue.json contains the dialogues.

- Each dialogue has a unique case index value in the json file, and is a list composed of the utterances in speaking order.
- Every utterance in the list contains the speaker, content, and annotated labels shown in data format.
- The list of the listener in the utterance contains all listeners in this utterance with their relation type.

The data format of dialogue.json is shown as follows.

```json
{
  case index:
    [
      {
        "speaker": speaker's name,
        "utterance": utterance,
        "listener":
          [
            {
              "name": listener's name,
              "relation": relation type between speaker and listener
              }, ...
          ],
            "emotion": speaker's emotion type
      }, ...
    ]
}
```

## metadata.json

- The metadata is given in metadata.json.
- The file defines all the emotion, relation types, and the sub-classes in the two perspectives, position, and field.
- The data format of metadata.json is shown as follows.

```json
{
   "relation":[
      "parent",
      ...
   ],
   "field":{
      "family":[
         "parent",
         ...
      ],
      "school":[
         "teacher",
         ...
      ],
      "company":[
         "boss",
         ...
      ],
      "others":[
         "couple",
         ...
      ]
   },
   "position":{
      "superior":[
         "parent",
         ...
      ],
      "peer":[
         "spouse",
         ...
      ],
      "inferior":[
         "child",
         ...
      ]
   },
   "emotion":[
      "fear",
      ...
   ]
}
```