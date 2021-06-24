#!/usr/bin/python
#
# Cityscapes labels
#

from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color'       , # The color of this label
] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'unlabeled'            , 0 ,        24 , 'void'            , 0       , False        , True         , (  0,  0,  0)),
#     Label(  'apple'            ,     25 ,        0 , 'doll'            , 3       , True         , False        , (227,114,170)),
#     Label(  'baseball'              ,26 ,        1 , 'glue'            , 7       , True         , False        , (225, 26,108)),
#     Label(  'banana'         ,       27 ,        2 , 'tool'             ,7       , True         , False        , (243,198,120)),
#     Label(  'eraser'           ,     28 ,        3 , 'doll'            , 5       , True         , False        , (228,127, 42)),
#     Label(  'tomatosoup'            ,29 ,        4 , 'doll'             ,4       , True         , True         , (169,228,233)),
#     Label(  'clamp'              ,   30 ,        5 , 'pack'             ,1       , True         , True         , ( 76,169,211)),
#     Label(  'cube'           ,       31 ,        6 , 'ball'             ,4       , True         , False        , ( 48, 96,166)),
#     Label(  'redcup'           ,     32 ,        7 , 'PET'            ,  3       , False        , True         , (121,173 ,65)),
#     Label(  'driver'              ,  33 ,        8 , 'tool'             ,2       , False        , True         , ( 66, 44,118)),
#     Label(  'boxdiget'             , 34 ,        9 , 'snack'            ,1       , False        , True         , (181, 58, 58)),
#     Label(  'diget'        ,         35 ,       10 , 'tape'             ,1       , False        , True         , (227, 53, 36)),
#     Label(  'dumbel'          ,      36 ,       11 , 'tape'             ,1       , False        , True         , (247,196, 20)),
#     Label(  'gotica'           ,     37 ,       12 , 'cup'             , 1       , False        , True         , (126, 73,155)),
#     Label(  'fork'                 , 38 ,       13 , 'dish'             ,1       , False        , True         , ( 49, 71,156)),
#     Label(  'lock'                 , 39,        14 , 'bowl'             ,1       , False        , True         , (116, 41, 66)),
#     Label(  'ladle'               ,  40 ,       15 , 'can'             , 1       , False        , True         , (255, 32, 67)),
#     Label(  'orange'               , 41 ,       16 , 'can'             , 1       , False        , True         , (128, 10, 200)),
#     Label(  'spam',                  42,        17,  'can',              1,        False,         True,          (60, 80, 160)),
#     Label(  'woodglue',              43,        18,  'can',              1,        False,         True,          (90, 50, 130)),
#     Label(  'vitamin_water',         44,        19,  'can',              1,        False,         True,          (150, 30, 100)),
#     Label(  'rubberduck',            45,        20,  'can',              1,        False,         True,          (200, 200, 10)),
#     Label(  'sauce',                 46,        21,  'can',              1,        False,         True,          (10, 170, 130)),
#     Label(  'scoop',                 47,        22,  'can',              1,        False,         True,          (40, 150, 230)),
#     Label(  'spoon',                 48,        23,  'can',              1,        False,         True,          (230, 40, 140))
# ]

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label( 'unlabeled',              0 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0)),
    Label(  'Drawer_Dark'           ,     2 ,       12 , 'cup'             , 1       , False        , True         , (126, 73,155)),
    Label(  'Bin_white'                 , 3 ,       13 , 'dish'             ,1       , False        , True         , ( 49, 71,156)),
    Label(  'Bin_stan'                 , 4,        14 , 'bowl'             ,1       , False        , True         , (116, 41, 66)),
    Label('Eraser_big', 18, 2, 'glue', 7, True, False, (225, 26, 108)),
    Label('Usb_Big', 20, 3, 'tool', 7, True, False, (243, 198, 120)),
    Label( 'Stapler_pink',               25 ,        1 , 'doll'            , 3       , True         , False        , (227,114,170)),
#     Label(  'gotica'           ,     37 ,       12 , 'cup'             , 1       , False        , True         , (126, 73,155)),
#     Label(  'fork'                 , 38 ,       13 , 'dish'             ,1       , False        , True         , ( 49, 71,156)),
#     Label(  'lock'                 , 39,        14 , 'bowl'             ,1       , False        , True         , (116, 41, 66)),
#     Label(  'ladle'               ,  40 ,       15 , 'can'             , 1       , False        , True         , (255, 32, 67)),
#     Label(  'orange'               , 41 ,       16 , 'can'             , 1       , False        , True         , (128, 10, 200)),
#     Label(  'spam',                  42,        17,  'can',              1,        False,         True,          (60, 80, 160)),
#     Label(  'woodglue',              43,        18,  'can',              1,        False,         True,          (90, 50, 130)),
#     Label(  'vitamin_water',         44,        19,  'can',              1,        False,         True,          (150, 30, 100)),
#     Label(  'rubberduck',            45,        20,  'can',              1,        False,         True,          (200, 200, 10)),
#     Label(  'sauce',                 46,        21,  'can',              1,        False,         True,          (10, 170, 130)),
#     Label(  'scoop',                 47,        22,  'can',              1,        False,         True,          (40, 150, 230)),
#     Label(  'spoon',                 48,        23,  'can',              1,        False,         True,          (230, 40, 140))
]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

#--------------------------------------------------------------------------------
# Assure single instance name
#--------------------------------------------------------------------------------

# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None
def assureSingleInstanceName( name ):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of cityscapes labels:")
    print("")
    print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval' ))
    print("    " + ('-' * 98))
    for label in labels:
        print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval ))
    print("")

    print("Example usages:")

    # Map from name to label
    name = 'car'
    id   = name2label[name].id
    print("ID of label '{name}': {id}".format( name=name, id=id ))

    # Map from ID to label
    category = id2label[id].category
    print("Category of label with ID '{id}': {category}".format( id=id, category=category ))

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print("Name of label with trainID '{id}': {name}".format( id=trainId, name=name ))
