def application_set_config():
    config = {
        "version": "v1",
        "config": {
            "visState": {
                "filters": [],
                "layers": [
                    {
                        "id": "okt80tr",
                        "type": "hexagonId",
                        "config": {
                            "dataId": "application",
                            "label": "H3_BLOCKS",
                            "color": [18, 147, 154],
                            "columns": {"hex_id": "H3_BLOCKS"},
                            "isVisible": True,
                            "visConfig": {
                                "opacity": 0.8,
                                "colorRange": {
                                    "name": "ColorBrewer RdBu-11",
                                    "type": "diverging",
                                    "category": "ColorBrewer",
                                    "colors": [
                                        "#053061",
                                        "#2166ac",
                                        "#4393c3",
                                        "#92c5de",
                                        "#d1e5f0",
                                        "#f7f7f7",
                                        "#fddbc7",
                                        "#f4a582",
                                        "#d6604d",
                                        "#b2182b",
                                        "#67001f",
                                    ],
                                    "reversed": True,
                                },
                                "coverage": 1,
                                "enable3d": False,
                                "sizeRange": [0, 500],
                                "coverageRange": [0, 1],
                                "elevationScale": 5,
                            },
                            "hidden": False,
                            "textLabel": [
                                {
                                    "field": None,
                                    "color": [255, 255, 255],
                                    "size": 18,
                                    "offset": [0, 0],
                                    "anchor": "start",
                                    "alignment": "center",
                                }
                            ],
                        },
                        "visualChannels": {
                            "colorField": {"name": "PERCENTILE", "type": "real"},
                            "colorScale": "quantize",
                            "sizeField": None,
                            "sizeScale": "linear",
                            "coverageField": None,
                            "coverageScale": "linear",
                        },
                    }
                ],
                "interactionConfig": {
                    "tooltip": {
                        "fieldsToShow": {
                            "application": [
                                {"name": "H3_BLOCKS", "format": None},
                                {"name": "SCORE", "format": None},
                                {"name": "PERCENTILE", "format": None},
                            ]
                        },
                        "compareMode": False,
                        "compareType": "absolute",
                        "enabled": True,
                    },
                    "brush": {"size": 0.5, "enabled": False},
                    "geocoder": {"enabled": False},
                    "coordinate": {"enabled": False},
                },
                "layerBlending": "normal",
                "splitMaps": [],
                "animationConfig": {"currentTime": None, "speed": 1},
            },
            "mapState": {
                "bearing": 0,
                "dragRotate": False,
                "latitude": 39.9984530968717,
                "longitude": -116.45639316147737,
                "pitch": 0,
                "zoom": 6,
                "isSplit": False,
            },
            "mapStyle": {
                "styleType": "dark",
                "topLayerGroups": {},
                "visibleLayerGroups": {
                    "label": True,
                    "road": True,
                    "border": False,
                    "building": True,
                    "water": True,
                    "land": True,
                    "3d building": False,
                },
                "threeDBuildingColor": [9.665468314072013, 17.18305478057247, 31.1442867897876],
                "mapStyles": {
                    "7v0e42s": {
                        "accessToken": None,
                        "custom": True,
                        "icon": "https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v9/static/-122.3391,37.7922,9,0,0/400x300?access_token=pk.eyJ1IjoidWNmLW1hcGJveCIsImEiOiJja2tyMjNhcWIwc29sMnVzMThoZ3djNXhzIn0._hfBNwCD7pCU7RAMOq6vUQ&logo=False&attribution=False",
                        "id": "7v0e42s",
                        "label": "Mapbox Satellite Streets",
                        "url": "mapbox://styles/mapbox/satellite-streets-v9",
                    }
                },
            },
        },
    }
    return config
