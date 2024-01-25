local function copyResource(file)
    path = quarto.utils.resolvePath(file)
    quarto.doc.addFormatResource(path)
end

function Header(el)
    if not el.attributes['background-image'] then
        el.attributes['background-size'] = 'contain'
    end

    return el
end
