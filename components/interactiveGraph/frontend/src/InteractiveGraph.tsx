import {
  Streamlit,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib"
import React, { useEffect, useRef, useState } from "react"
import { graphviz } from "d3-graphviz"
import { selectAll, select } from "d3"
import { v4 as uuidv4 } from "uuid"

type nodeClickData = {
  clickId: string
  nodeId: string
}

type edgeClickData = {
  edgeClickId: string
  source: string
  target: string
}

const InteractiveGraph: React.FC<ComponentProps> = ({ args }) => {
  const dot_source = args["graphviz_string"]
  const key = args["key"]
  const height: number = args["height"]
  const div_ref: React.Ref<HTMLDivElement> = useRef<HTMLDivElement>(null)
  const [nodeClickData, setNodeClickData] = useState<nodeClickData>({
    clickId: "",
    nodeId: "",
  })
  const [edgeClickData, setEdgeClickData] = useState<edgeClickData>({
    edgeClickId: "",
    source: "",
    target: "",
  })
  const [width, setWidth] = useState(0)

  const [isRendering, setIsRendering] = useState(true)

  function resetGraph() {
    graphviz(".graph").fit(true).resetZoom()
  }

  function bindAfterRender() {
    resetGraph()
    selectAll(".node").on("click", (event) => {
      event.preventDefault()
      const node_id = event.target.__data__.parent.key
      console.log(node_id)
      setNodeClickData({
        clickId: uuidv4(),
        nodeId: node_id,
      })
    })
    selectAll(".edge").on("click", (event) => {
      event.preventDefault()
      const titleText = select(event.currentTarget).select("title").text()
      const [src, tgt] = titleText.split("->")
      setEdgeClickData({
        edgeClickId: uuidv4(),
        source: src,
        target: tgt,
      })
    })
    setIsRendering(false)
  }

  useEffect(() => {
    Streamlit.setFrameHeight(height)

    const onResize = () => {
      if (div_ref.current) {
        setWidth(div_ref.current.clientWidth)
      }
    }

    onResize()
    window.addEventListener("resize", onResize)

    // cleanup
    return () => {
      window.removeEventListener("resize", onResize)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (width === 0 || height === 0) return

    const graphviz_instance = graphviz(".graph")

    const render_timeout = setTimeout(() => {
      setIsRendering(true)
      graphviz_instance
        .width(width)
        .height(height)
        .fit(true)
        .zoomScaleExtent([0.1, 100])
        .on("end", bindAfterRender)
        .renderDot(dot_source)
    }, 100)

    return () => {
      clearTimeout(render_timeout)
      // destroy old graphviz rendering instance
      if (graphviz_instance) {
        const any_instance = graphviz_instance as any
        any_instance.destroy()
        // remove old div content
        const any_graph_div = select(".graph").node() as any
        any_graph_div.innerHTML = ""
      }
    }

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dot_source, width, height])

  useEffect(() => {
    Streamlit.setComponentValue(nodeClickData)
  }, [nodeClickData])

  useEffect(() => {
    Streamlit.setComponentValue(edgeClickData)
  }, [edgeClickData])

  return (
    <div
      id={key}
      ref={div_ref}
      style={{
        position: "absolute",
        height: "100%",
        width: "100%",
        backgroundColor: "white",
      }}
    >
      <div
        className="graph"
        style={{
          position: "absolute",
          height: "100%",
          width: "100%",
        }}
      ></div>
      {isRendering ? (
        <div
          style={{
            position: "absolute",
            display: "flex",
            height: "100%",
            width: "100%",
            backgroundColor: "#FEFEFE",
            alignItems: "center",
            justifyContent: "center",
            flexDirection: "column",
          }}
        >
          <p style={{ fontSize: "2em", color: "black" }}>
            The Graph is being rendered. For larger graphs this may take a
            while.
          </p>
          <p style={{ fontSize: "2em", color: "black" }}>
            Try changing the parameters to reduce the graph size.
          </p>
        </div>
      ) : (
        <button
          onClick={resetGraph}
          style={{
            position: "absolute",
            top: 0,
            right: 0,
            backgroundColor: "#F0F0F0",
            borderRadius: "0.5rem",
            minHeight: "38px",
            padding: "0.25 rem 0.75rem",
            margin: "0.25rem",
            border: "none",
            outline: "none",
          }}
        >
          Reset
        </button>
      )}
    </div>
  )
}

export default withStreamlitConnection(InteractiveGraph)
